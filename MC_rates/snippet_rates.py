
    def calc_merger_rates(self,
                          time_bins: int | NDArray,
                          zmin: float = 0.0, zmax: float = 0.1) -> tuple[Quantity, Quantity, Quantity, Quantity, pd.DataFrame]:
        '''
        Calculate DCO merger rates for the sample following Dominik et al. 2013 Eq. 16. 
        ### Parameters:
        - time_bins: int | NDArray
        - zmin: float = 0.0
        - zmax: float = 0.1
        '''
        if (zmin < 0 or zmax < 0):
            raise ValueError('Redshift must be positive!')
        
        if isinstance(time_bins, int):
            if time_bins < 50:
                print(f"Warning: {time_bins} is not a lot of bins. You should probably use more.")
            ti = self.comoving_time[0]
            tf = self.comoving_time[-1]
            time_bins = np.linspace(ti, tf, time_bins).to(u.Myr)
        elif (isinstance(time_bins, NDArray) or isinstance(time_bins, list)):
            if not isinstance(time_bins, Quantity):
                raise ValueError("Please provide time units!")
        
        n: int = time_bins.shape[0] - 1
        
        data: dict[str:NDArray] = {
            "t_i": np.zeros(shape=n),
            "t_f": np.zeros(shape=n),
            "t_center": np.zeros(shape=n),
            "z_center": np.zeros(shape=n),
            "differential_comoving_volume": np.zeros(shape=n),
            "bbh_rate": np.zeros(shape=n),
            "bhns_rate": np.zeros(shape=n),
            "bns_rate": np.zeros(shape=n),
            "total_rate": np.zeros(shape=n)
        }

        for i in range(time_bins[:-1].shape[0]):

            t_i = time_bins[i]
            t_f = time_bins[i+1]
            t_at_center: Quantity = np.mean(time_bins[i:i+2]).to(u.Myr)
            z_at_center: Quantity = z_at_value(self.cosmology.age, t_at_center)
            
            if (z_at_center < zmin or z_at_center > zmax):
                continue
            
            total_rate = np.zeros(shape=len(self.bins)) * (u.yr ** -1 * u.Gpc ** -3)
            bbh_rate = total_rate.copy()
            bhns_rate = total_rate.copy()
            bns_rate = total_rate.copy()
            
            for j, Zbin in enumerate(self.bins):
                
                f_corr: float = Zbin.imf_f_corr
                Msim: Quantity = Zbin.simulated_mass 
                mergers: pd.DataFrame = Zbin.mergers
                valid_mergers: pd.DataFrame = mergers[(mergers.t_delay * u.Myr) < t_at_center]
                
                bbh_systems: NDArray = valid_mergers[(valid_mergers.kstar_1 == 14) & (valid_mergers.kstar_2 == 14)].index
                bns_systems: NDArray = valid_mergers[(valid_mergers.kstar_1 == 13) & (valid_mergers.kstar_2 == 13)].index
                bhns_systems: NDArray = valid_mergers[((valid_mergers.kstar_1 == 13) & (valid_mergers.kstar_2 == 14))\
                    | ((valid_mergers.kstar_1 == 14) & (valid_mergers.kstar_2 == 13))].index
                
                
                t_form: NDArray = t_at_center - (valid_mergers.t_delay.values * u.Myr)
                SFR_per_system: NDArray = np.interp(t_form, self.comoving_time, self.fractional_SFR_at_met[j,:])
                
                # Dominik+13 Eq. 16
                merger_rate: pd.Series = pd.Series(
                    ((n ** -1) * (f_corr / Msim) * SFR_per_system ).to(u.yr ** -1 * u.Gpc ** -3), index=valid_mergers.index)
                
                total_rate[j] = merger_rate.sum()
                bbh_rate[j] = merger_rate.loc[bbh_systems].sum() * (u.yr ** -1 * u.Gpc ** -3)
                bhns_rate[j] = merger_rate.loc[bhns_systems].sum() * (u.yr ** -1 * u.Gpc ** -3)
                bns_rate[j] = merger_rate.loc[bns_systems].sum() * (u.yr ** -1 * u.Gpc ** -3)

            data["t_i"][i] = t_i.value
            data["t_f"][i] = t_f.value
            data["t_center"][i] = t_at_center.value
            data["z_center"][i] = z_at_center
            data["differential_comoving_volume"][i] = np.nan
            data["bbh_rate"][i] = bbh_rate.sum().value
            data["bhns_rate"][i] = bhns_rate.sum().value
            data["bns_rate"][i] = bns_rate.sum().value
            data["total_rate"][i] = total_rate.sum().value
        
        tot_rate = data["total_rate"].sum() * (u.yr ** -1 * u.Gpc ** -3)
        bbh = data["bbh_rate"].sum() * (u.yr ** -1 * u.Gpc ** -3)
        bhns = data["bhns_rate"].sum() * (u.yr ** -1 * u.Gpc ** -3)
        bns = data["bns_rate"].sum() * (u.yr ** -1 * u.Gpc ** -3)
        df = pd.DataFrame(data)
        
        return tot_rate, bbh, bhns, bns, df

    def calc_MC_rates(self: MCParams,
                    n: int = 100, seed: int = 0, **kwargs) -> pd.DataFrame:
        '''
        Calculate rates by taking a monte-carlo sum with binaries from `sampler`.
        ### Parameters:
        - n: int - the number of monte-carlo samples to draw from each system
        - seed: int - the pseudorandom seed 
        #### Kwargs:
        - dt: Quantity - the time interval to determine comoving time bins
        - detectability_function: function - the function accounting for detector effects
        ### Returns:
        - DataFrame - a dataframe of MC samples, weights, and detectability
        '''
        bins: list[Model] = self.bins
        
        dt: Quantity = kwargs.get("dt", 100 * u.Myr)
        detectability_function: function = kwargs.get("detectability_function", trivial_Pdet)
        time_bins: NDArray = np.arange(self.comoving_time[0].value, self.comoving_time[-1].value, dt.value)
        
        dicts = []
        
        for j_Z, m in enumerate(bins):
            metallicity: float = m.metallicity
            mergers: pd.DataFrame = m.mergers
            n_k: int = mergers.index.shape[0]
            
            # trying with weighted SFR
            fracSFR_pdf_at_met = self.weighted_SFR[j_Z,:]
            
            # get system statistics
            bin_num = mergers.bin_num.values
            mass_1 = mergers.mass_1.values
            mass_2 = mergers.mass_2.values
            kstar_1 = mergers.kstar_1.values
            kstar_2 = mergers.kstar_2.values
            t_delay = mergers.t_delay.values
            
            # sample formation times with ITM
            harvest = self._inverse_transform_sample(
                (n_k, n), fracSFR_pdf_at_met, self.comoving_time, seed)
            
            t_max = self.comoving_time.max().to(u.Myr).value
            sample_list = np.array([], dtype=float)
            num_valid_samples = np.zeros(n_k, dtype=np.int64)
            
            num_valid_samples = np.ones(n_k, dtype=np.int64) * n
            sample_list = np.reshape(harvest, n_k*n)
                
            columns = {
                "bin_num": np.repeat(bin_num, num_valid_samples),
                "metallicity": np.ones(sample_list.shape[0], dtype=float) * metallicity,
                "mass_1": np.repeat(mass_1, num_valid_samples),
                "mass_2": np.repeat(mass_2, num_valid_samples),
                "kstar_1": np.repeat(kstar_1, num_valid_samples),
                "kstar_2": np.repeat(kstar_2, num_valid_samples),
                "t_delay": np.repeat(t_delay, num_valid_samples),
                "t_form": sample_list,
                "z_form": np.interp(sample_list, self.comoving_time.value, self.redshift),
                "t_merge": sample_list + np.repeat(t_delay, num_valid_samples),
                "z_merge": np.interp(sample_list+np.repeat(t_delay, num_valid_samples), self.comoving_time.value, self.redshift, right=np.nan),
                }
            
            columns["P_det"] = detectability_function(columns)
            columns["Rate"] = self._calc_intrinsic_rates(j_Z, columns, n, dt, time_bins)
            
            dicts.append(columns)
            
        final_data = {a:None for a in dicts[0].keys()}
        for di in dicts:
            for k, v in di.items():
                if final_data[k] is None:
                    final_data[k] = v
                else:
                    final_data[k] = np.append(final_data[k], v)
        
        output = pd.DataFrame(final_data)
        return output

    def _calc_intrinsic_rates(self: MCParams,
                    j_Z: int, data: dict[str:NDArray], num_draws: int, dt: Quantity, time_bins: NDArray) -> NDArray:
        '''
        We estimate intrinsic merger rates by binning our sampled events by merger time and multiplying by
        the star formation history SFH(z,Z) and population correction factors.
        For an explanation of the SFH see Bavera et al. 2020.
        '''
        dt = dt.to(u.Myr)
        i_t: int = time_bins.shape[0] - 1
        output = np.zeros(data["t_form"].shape[0]) * (u.yr ** -1 * u.Gpc ** -3)
        
        for i in range(i_t):
            t0, t1 = time_bins[i], time_bins[i+1]
            idx = (data["t_merge"] >= t0) & (data["t_merge"] < t1)
            
            z_form = data["z_form"][idx]
            
            # star formation history
            SFR_at_z: NDArray =  np.interp(z_form, self.redshift[::-1], self.SFR_at_z[::-1]) # TODO
            met_frac_at_z: float = np.interp(z_form, self.redshift[::-1], self.fractional_SFR_at_met[j_Z,:][::-1]) # TODO
            SFH_jz = (SFR_at_z * met_frac_at_z).to(u.Msun * u.yr ** -1 * u.Gpc ** -3)
            fcorr_SFR_fracs = np.interp(z_form, self.redshift[::-1], self.fcorr_SFR_fracs[::-1]) # TODO
            
            # intrinsic merger rate -- see Dominik+13 eq. 16             
            rate = self.bins[j_Z].imf_f_corr * (1/num_draws) * fcorr_SFR_fracs *\
                SFH_jz * (self.bins[j_Z].total_star_mass ** -1)# * \
                            
            rate = rate.to(u.yr ** -1 * u.Gpc ** -3)
            output[idx] = rate
        
        return output
    
    @staticmethod
    def _inverse_transform_sample(shape: tuple[int,int],
                                fracSFR_pdf_at_met: NDArray, comoving_time: NDArray, seed: int = 0) -> NDArray:
        '''
        Use inverse transform sampling to sample continuously in time.
        Specifically, this sampler accepts a metallicity bin and calculates the CDF of stars of that
        metallicity forming across cosmic time; it then draws a sample of formation times for stars of that
        particular metallicity.
        **NOTE:** This does not account for the absolute SFR at a given time, only the metallicity distribution! 
        ### Parameters:
        - shape: tuple[int, int] - the shape of the sample to draw
        - j_Z: int - the index of the metallicity at which to sample
        - fracSFR_pdf_at_met: NDArray - the fractional SFR at metallicity j_Z over time
        - comoving_time: the comoving time values over which to interpolate
        - seed: int - the pseudorandom seed
        ### Returns:
        - NDArray - the pseudorandom sample
        '''
        Z_pdf = fracSFR_pdf_at_met
        norm_pdf = (Z_pdf / np.sum(Z_pdf)) # normalize
        cdf = np.cumsum(norm_pdf)
        #print(cdf[-1])

        gen = np.random.default_rng(seed=seed)
        choices = gen.random(size=shape)
        harvest = np.interp(choices, cdf, comoving_time.value)
        
        return harvest
