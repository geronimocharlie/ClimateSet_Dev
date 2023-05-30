# Supported Model sources

MODEL_SOURCES = {
    "NorESM2-LM": {"node_link": "https://esgf-data.dkrz.de/esg-search", "center": "NCC"},
    "CanESM5" : {"node_link": "http://esgf-node.llnl.gov/esg-search/", "center": "CCCma"}
}


OPENID = "https://esgf-node.llnl.gov/esgf-idp/openid/causalpaca" #https://esgf-data.dkrz.de/esgf-idp/openid/causalpaca"
PASSWORD = "Causalpaca.42"


VAR_SOURCE_LOOKUP = {
    "model": [
        "ztp",
        "zsatcalc",
        "zsatarag",
        "zostoga",
        "zossq",
        "zos",
        "zoocos",
        "zooc",
        "zo2min",
        "zhalfo",
        "zg500",
        "zg1000",
        "zg100",
        "zg10",
        "zg",
        "zfullo",
        "wtd",
        "wo",
        "wmo",
        "wfonocorr",
        "wfo",
        "wetss",
        "wetso4",
        "wetso2",
        "wetlandFrac",
        "wetlandCH4",
        "wetbc",
        "wap500",
        "wap",
        "vsf",
        "volo",
        "volcello",
        "vo",
        "vmo",
        "vegHeight",
        "va",
        "uo",
        "umo",
        "ua",
        "tslsi",
        "tsl",
        "ts",
        "tran",
        "tossq",
        "tosga",
        "tos",
        "tob",
        "thkcello",
        "thetaot700",
        "thetaot300",
        "thetaot2000",
        "thetaot",
        "thetaoga",
        "thetao",
        "tgs",
        "tcs",
        "tauvo",
        "tauv",
        "tauuo",
        "tauu",
        "tasmin",
        "tasmax",
        "tas",
        "talkos",
        "talknat",
        "talk",
        "ta850",
        "ta700",
        "ta500",
        "ta",
        "t20d",
        "spco2",
        "sossq",
        "sosga",
        "sos",
        "sootsn",
        "somint",
        "soga",
        "sob",
        "so2",
        "so",
        "snw",
        "sndmasswindrif",
        "sndmasssnf",
        "sndmasssi",
        "sndmassmelt",
        "snd",
        "snc",
        "sivols",
        "sivoln",
        "sivol",
        "siv",
        "siu",
        "sitimefrac",
        "sithick",
        "sitemptop",
        "sitempsnic",
        "sitempbot",
        "sistryubot",
        "sistrydtop",
        "sistrxubot",
        "sistrxdtop",
        "sispeed",
        "sisnthick",
        "sisnmass",
        "sisnhc",
        "sisnconc",
        "sirdgthick",
        "sirdgconc",
        "sipr",
        "sios",
        "simpconc",
        "simass",
        "siitdthick",
        "siitdsnthick",
        "siitdsnconc",
        "siitdconc",
        "sihc",
        "siforcetilty",
        "siforcetiltx",
        "siforceintstry",
        "siforceintstrx",
        "siforcecorioly",
        "siforcecoriolx",
        "siflswutop",
        "siflswdtop",
        "siflswdbot",
        "siflsensupbot",
        "siflsenstop",
        "sifllwutop",
        "sifllwdtop",
        "sifllatstop",
        "siflfwdrain",
        "siflfwbot",
        "siflcondtop",
        "siflcondbot",
        "sifb",
        "siextents",
        "siextentn",
        "sidmasstrany",
        "sidmasstranx",
        "sidmassth",
        "sidmasssi",
        "sidmassmelttop",
        "sidmassmeltbot",
        "sidmasslat",
        "sidmassgrowthwat",
        "sidmassgrowthbot",
        "sidmassevapsubl",
        "sidmassdyn",
        "sidivvel",
        "sidconcth",
        "sidconcdyn",
        "siconc",
        "sicompstren",
        "siarean",
        "siage",
        "si",
        "sftof",
        "sftlf",
        "sftgif",
        "sfdsi",
        "sfcWind",
        "sf6",
        "rtmt",
        "rsutcsaf",
        "rsutcs",
        "rsutaf",
        "rsut",
        "rsuscs",
        "rsus",
        "rsntds",
        "rsdt",
        "rsdsdiff",
        "rsdscs",
        "rsds",
        "rlutcsaf",
        "rlutcs",
        "rlutaf",
        "rlut",
        "rlus",
        "rldscs",
        "rlds",
        "rh",
        "reffclwtop",
        "ra",
        "rMaint",
        "rGrowth",
        "qgwr",
        "pso",
        "psl",
        "ps",
        "prw",
        "prveg",
        "prsn",
        "prra",
        "prc",
        "pr",
        "ppos",
        "pp",
        "popos",
        "pop",
        "ponos",
        "pon",
        "po4os",
        "po4",
        "phynos",
        "phyn",
        "phyfeos",
        "phyfe",
        "phyc",
        "phos",
        "phnat",
        "phalf",
        "ph",
        "pfull",
        "pctisccp",
        "pbo",
        "orog",
        "opottempmint",
        "oh",
        "od870aer",
        "od550ss",
        "od550so4",
        "od550oa",
        "od550lt1aer",
        "od550dust",
        "od550csaer",
        "od550bc",
        "od550aerh2o",
        "od550aer",
        "od440aer",
        "obvfsq",
        "o3",
        "o2satos",
        "o2sat",
        "o2os",
        "o2min",
        "o2",
        "nppWood",
        "nppRoot",
        "nppLeaf",
        "npp",
        "no3os",
        "no3",
        "nep",
        "nbp",
        "nVeg",
        "nStem",
        "nSoil",
        "nRoot",
        "nMineralNO3",
        "nMineralNH4",
        "nMineral",
        "nLitter",
        "nLeaf",
        "nLand",
        "n2oglobal",
        "msftmzmpa",
        "msftmz",
        "msftmrhompa",
        "msftmrho",
        "msftbarot",
        "mrtws",
        "mrsos",
        "mrsol",
        "mrso",
        "mrsll",
        "mrsfl",
        "mrros",
        "mrrob",
        "mrro",
        "mrlso",
        "mrfso",
        "mmrss",
        "mmrsoa",
        "mmrso4",
        "mmrpm2p5",
        "mmrpm1",
        "mmroa",
        "mmrdust",
        "mmrbc",
        "mmraerh2o",
        "mlotstsq",
        "mlotstmin",
        "mlotstmax",
        "mlotst",
        "mfo",
        "masso",
        "masscello",
        "lwsnl",
        "lwp",
        "loadss",
        "loaddust",
        "lai",
        "isop",
        "intpp",
        "intpoc",
        "intpn2",
        "intdoc",
        "intdic",
        "huss",
        "hus",
        "hurs",
        "hur",
        "hfy",
        "hfx",
        "hfss",
        "hfls",
        "hfds",
        "hfbasinpmdiff",
        "hfbasinpmadv",
        "hfbasinpadv",
        "hfbasin",
        "gpp",
        "fsitherm",
        "froc",
        "frn",
        "friver",
        "fric",
        "frfe",
        "ficeberg",
        "fgo2",
        "fgdms",
        "fgco2nat",
        "fgco2",
        "fVegLitterSenescence",
        "fVegLitterMortality",
        "fVegLitter",
        "fNup",
        "fNnetmin",
        "fNloss",
        "fNleach",
        "fNgasNonFire",
        "fNgasFire",
        "fNgas",
        "fNfert",
        "fNdep",
        "fNProduct",
        "fNOx",
        "fN2O",
        "fLuc",
        "fLitterFire",
        "fHarvestToProduct",
        "fHarvest",
        "fFireNat",
        "fFire",
        "fDeforestToProduct",
        "fBNF",
        "evspsblveg",
        "evspsblsoi",
        "evspsbl",
        "evs",
        "esn",
        "es",
        "epsi100",
        "epp100",
        "epn100",
        "epfe100",
        "epcalc100",
        "epc100",
        "emivoc",
        "emiss",
        "emiso4",
        "emiso2",
        "emioa",
        "emiisop",
        "emidust",
        "emidms",
        "emibvoc",
        "emibc",
        "ec",
        "dryso4",
        "dryso2",
        "drybc",
        "dpco2",
        "dmsos",
        "dms",
        "dmlt",
        "dissocos",
        "dissoc",
        "dissicos",
        "dissicnat",
        "dissic",
        "dfeos",
        "dfe",
        "detocos",
        "detoc",
        "deptho",
        "cod",
        "co3satcalcos",
        "co3satcalc",
        "co3sataragos",
        "co3satarag",
        "co3os",
        "co3nat",
        "co3",
        "co2mass",
        "co2",
        "clwvi",
        "clwmodis",
        "clw",
        "cltmodis",
        "cltisccp",
        "cltcalipso",
        "clt",
        "clmcalipso",
        "cllcalipso",
        "clivi",
        "climodis",
        "cli",
        "clhcalipso",
        "cl",
        "chlos",
        "chl",
        "chepsoa",
        "ch4global",
        "cfc12global",
        "cfc12",
        "cfc11global",
        "cfc11",
        "cdnc",
        "cct",
        "ccn",
        "ccb",
        "calcos",
        "calc",
        "cWood",
        "cVeg",
        "cStem",
        "cSoilSlow",
        "cSoilMedium",
        "cSoilFast",
        "cSoilAbove1m",
        "cSoil",
        "cRoot",
        "cMisc",
        "cLitter",
        "cLeaf",
        "cLand",
        "cCwd",
        "bsios",
        "bsi",
        "bldep",
        "bfeos",
        "bfe",
        "basin",
        "ares",
        "areacello",
        "areacella",
        "albisccp",
        "airmass",
        "agessc",
        "abs550aer",
    ],
    "raw": [
        "years",
        "year_weight",
        "year_fr",
        "wlenbinsize",
        "wlen_bnds",
        "wlen",
        "wfo",
        "wetnoy",
        "wetnhx",
        "water_vapor",
        "vos",
        "volume_density",
        "vo",
        "vmro3",
        "vas",
        "urban_to_secdn",
        "urban_to_secdf",
        "urban_to_range",
        "urban_to_pastr",
        "urban_to_c4per",
        "urban_to_c4ann",
        "urban_to_c3per",
        "urban_to_c3nfx",
        "urban_to_c3ann",
        "urban",
        "uos",
        "uo",
        "uas",
        "tsi",
        "ts",
        "total_solar_irradiance",
        "tosbcs",
        "tos",
        "thetao",
        "theta",
        "temp_level",
        "temp_layer",
        "tauv",
        "tauu",
        "tas",
        "surface_temperature",
        "surface_emissivity",
        "surface_albedo",
        "sst",
        "ssn",
        "ssi",
        "ssa550",
        "sos",
        "solar_zenith_angle",
        "so2f2_SH",
        "so2f2_NH",
        "so2f2_GM",
        "so",
        "sithick",
        "sig_lon_W",
        "sig_lon_E",
        "sig_lat_W",
        "sig_lat_E",
        "siconcbcs",
        "siconca",
        "siconc",
        "sftof",
        "sftflf",
        "sf6_SH",
        "sf6_NH",
        "sf6_GM",
        "secyf_harv",
        "secyf_bioh",
        "secnf_harv",
        "secnf_bioh",
        "secmf_harv",
        "secmf_bioh",
        "secmb",
        "secma",
        "secdn_to_urban",
        "secdn_to_secdf",
        "secdn_to_range",
        "secdn_to_pastr",
        "secdn_to_c4per",
        "secdn_to_c4ann",
        "secdn_to_c3per",
        "secdn_to_c3nfx",
        "secdn_to_c3ann",
        "secdn",
        "secdf_to_urban",
        "secdf_to_secdn",
        "secdf_to_range",
        "secdf_to_pastr",
        "secdf_to_c4per",
        "secdf_to_c4ann",
        "secdf_to_c3per",
        "secdf_to_c3nfx",
        "secdf_to_c3ann",
        "secdf",
        "scph",
        "scnum",
        "sad_of_big_particles",
        "sad",
        "rsds",
        "rndwd",
        "rmean",
        "rlds",
        "range_to_urban",
        "range_to_secdn",
        "range_to_secdf",
        "range_to_pastr",
        "range_to_c4per",
        "range_to_c4ann",
        "range_to_c3per",
        "range_to_c3nfx",
        "range_to_c3ann",
        "range",
        "ptbio",
        "psl",
        "prsn",
        "prra",
        "profile_weight",
        "primn_to_urban",
        "primn_to_secdf",
        "primn_to_range",
        "primn_to_pastr",
        "primn_to_c4per",
        "primn_to_c4ann",
        "primn_to_c3per",
        "primn_to_c3nfx",
        "primn_to_c3ann",
        "primn_harv",
        "primn_bioh",
        "primn",
        "primf_to_urban",
        "primf_to_secdn",
        "primf_to_range",
        "primf_to_pastr",
        "primf_to_c4per",
        "primf_to_c4ann",
        "primf_to_c3per",
        "primf_to_c3nfx",
        "primf_to_c3ann",
        "primf_harv",
        "primf_bioh",
        "primf",
        "pressure",
        "pres_level",
        "pres_layer",
        "pr",
        "plume_number",
        "plume_lon",
        "plume_lat",
        "plume_feature",
        "percentage_TEMF",
        "percentage_SAVA",
        "percentage_PEAT",
        "percentage_DEFO",
        "percentage_BORF",
        "percentage_AGRI",
        "pastr_to_urban",
        "pastr_to_secdn",
        "pastr_to_secdf",
        "pastr_to_range",
        "pastr_to_c4per",
        "pastr_to_c4ann",
        "pastr_to_c3per",
        "pastr_to_c3nfx",
        "pastr_to_c3ann",
        "pastr",
        "ozone",
        "oxygen_GM",
        "nitrous_oxide_SH",
        "nitrous_oxide_NH",
        "nitrous_oxide_GM",
        "nitrogen_GM",
        "nf3_SH",
        "nf3_NH",
        "nf3_GM",
        "mrro",
        "month",
        "mole_fraction_of_so2f2_in_air",
        "mole_fraction_of_sf6_in_air",
        "mole_fraction_of_nitrous_oxide_in_air",
        "mole_fraction_of_nf3_in_air",
        "mole_fraction_of_methyl_chloride_in_air",
        "mole_fraction_of_methyl_bromide_in_air",
        "mole_fraction_of_methane_in_air",
        "mole_fraction_of_hfc4310mee_in_air",
        "mole_fraction_of_hfc365mfc_in_air",
        "mole_fraction_of_hfc32_in_air",
        "mole_fraction_of_hfc245fa_in_air",
        "mole_fraction_of_hfc23_in_air",
        "mole_fraction_of_hfc236fa_in_air",
        "mole_fraction_of_hfc227ea_in_air",
        "mole_fraction_of_hfc152a_in_air",
        "mole_fraction_of_hfc143a_in_air",
        "mole_fraction_of_hfc134aeq_in_air",
        "mole_fraction_of_hfc134a_in_air",
        "mole_fraction_of_hfc125_in_air",
        "mole_fraction_of_hcfc22_in_air",
        "mole_fraction_of_hcfc142b_in_air",
        "mole_fraction_of_hcfc141b_in_air",
        "mole_fraction_of_halon2402_in_air",
        "mole_fraction_of_halon1301_in_air",
        "mole_fraction_of_halon1211_in_air",
        "mole_fraction_of_co2eq_in_air",
        "mole_fraction_of_chcl3_in_air",
        "mole_fraction_of_ch3ccl3_in_air",
        "mole_fraction_of_ch2cl2_in_air",
        "mole_fraction_of_cfc12eq_in_air",
        "mole_fraction_of_cfc12_in_air",
        "mole_fraction_of_cfc11eq_in_air",
        "mole_fraction_of_cfc11_in_air",
        "mole_fraction_of_cfc115_in_air",
        "mole_fraction_of_cfc114_in_air",
        "mole_fraction_of_cfc113_in_air",
        "mole_fraction_of_cf4_in_air",
        "mole_fraction_of_carbon_tetrachloride_in_air",
        "mole_fraction_of_carbon_dioxide_in_air",
        "mole_fraction_of_c_c4f8_in_air",
        "mole_fraction_of_c8f18_in_air",
        "mole_fraction_of_c7f16_in_air",
        "mole_fraction_of_c6f14_in_air",
        "mole_fraction_of_c5f12_in_air",
        "mole_fraction_of_c4f10_in_air",
        "mole_fraction_of_c3f8_in_air",
        "mole_fraction_of_c2f6_in_air",
        "methyl_chloride_SH",
        "methyl_chloride_NH",
        "methyl_chloride_GM",
        "methyl_bromide_SH",
        "methyl_bromide_NH",
        "methyl_bromide_GM",
        "methane_SH",
        "methane_NH",
        "methane_GM",
        "mask4resto_ipv_Nextrop",
        "mask4resto_ipv",
        "mask4resto_amv_trop",
        "mask4resto_amv_extrop",
        "mask4resto_amv",
        "lon_bounds",
        "licalvf",
        "lat_bounds",
        "kp",
        "is_biomass",
        "irrig_c4per",
        "irrig_c4ann",
        "irrig_c3per",
        "irrig_c3nfx",
        "irrig_c3ann",
        "ipv_index",
        "iprp",
        "iprm",
        "iprg",
        "icwtr",
        "huss",
        "hfds",
        "hfc4310mee_SH",
        "hfc4310mee_NH",
        "hfc4310mee_GM",
        "hfc365mfc_SH",
        "hfc365mfc_NH",
        "hfc365mfc_GM",
        "hfc32_SH",
        "hfc32_NH",
        "hfc32_GM",
        "hfc245fa_SH",
        "hfc245fa_NH",
        "hfc245fa_GM",
        "hfc23_SH",
        "hfc23_NH",
        "hfc23_GM",
        "hfc236fa_SH",
        "hfc236fa_NH",
        "hfc236fa_GM",
        "hfc227ea_SH",
        "hfc227ea_NH",
        "hfc227ea_GM",
        "hfc152a_SH",
        "hfc152a_NH",
        "hfc152a_GM",
        "hfc143a_SH",
        "hfc143a_NH",
        "hfc143a_GM",
        "hfc134aeq_SH",
        "hfc134aeq_NH",
        "hfc134aeq_GM",
        "hfc134a_SH",
        "hfc134a_NH",
        "hfc134a_GM",
        "hfc125_SH",
        "hfc125_NH",
        "hfc125_GM",
        "hcfc22_SH",
        "hcfc22_NH",
        "hcfc22_GM",
        "hcfc142b_SH",
        "hcfc142b_NH",
        "hcfc142b_GM",
        "hcfc141b_SH",
        "hcfc141b_NH",
        "hcfc141b_GM",
        "halon2402_SH",
        "halon2402_NH",
        "halon2402_GM",
        "halon1301_SH",
        "halon1301_NH",
        "halon1301_GM",
        "halon1211_SH",
        "halon1211_NH",
        "halon1211_GM",
        "gzdis",
        "gridcellarea",
        "gpbio",
        "gldis",
        "glat_bnds",
        "glat",
        "fulwd",
        "ftr_weight",
        "fstnf",
        "friver",
        "flood",
        "fill_flag",
        "fharv_c4per",
        "fharv_c3per",
        "fertl_c4per",
        "fertl_c4ann",
        "fertl_c3per",
        "fertl_c3nfx",
        "fertl_c3ann",
        "f107",
        "expt_label",
        "evspsbl",
        "drynoy",
        "drynhx",
        "delta13co2_in_air",
        "datasource",
        "crpbf_total",
        "crpbf_c4per",
        "crpbf_c4ann",
        "crpbf_c3per",
        "crpbf_c3nfx",
        "crpbf_c3ann",
        "combf",
        "co2eq_SH",
        "co2eq_NH",
        "co2eq_GM",
        "chcl3_SH",
        "chcl3_NH",
        "chcl3_GM",
        "ch3ccl3_SH",
        "ch3ccl3_NH",
        "ch3ccl3_GM",
        "ch2cl2_SH",
        "ch2cl2_NH",
        "ch2cl2_GM",
        "cfc12eq_SH",
        "cfc12eq_NH",
        "cfc12eq_GM",
        "cfc12_SH",
        "cfc12_NH",
        "cfc12_GM",
        "cfc11eq_SH",
        "cfc11eq_NH",
        "cfc11eq_GM",
        "cfc11_SH",
        "cfc11_NH",
        "cfc11_GM",
        "cfc115_SH",
        "cfc115_NH",
        "cfc115_GM",
        "cfc114_SH",
        "cfc114_NH",
        "cfc114_GM",
        "cfc113_SH",
        "cfc113_NH",
        "cfc113_GM",
        "cf4_SH",
        "cf4_NH",
        "cf4_GM",
        "ccode",
        "carea",
        "carbon_tetrachloride_SH",
        "carbon_tetrachloride_NH",
        "carbon_tetrachloride_GM",
        "carbon_monoxide_GM",
        "carbon_dioxide_SH",
        "carbon_dioxide_NH",
        "carbon_dioxide_GM",
        "calyear",
        "calmonth",
        "calday",
        "c_c4f8_SH",
        "c_c4f8_NH",
        "c_c4f8_GM",
        "c8f18_SH",
        "c8f18_NH",
        "c8f18_GM",
        "c7f16_SH",
        "c7f16_NH",
        "c7f16_GM",
        "c6f14_SH",
        "c6f14_NH",
        "c6f14_GM",
        "c5f12_SH",
        "c5f12_NH",
        "c5f12_GM",
        "c4per_to_urban",
        "c4per_to_secdn",
        "c4per_to_secdf",
        "c4per_to_range",
        "c4per_to_pastr",
        "c4per_to_c4ann",
        "c4per_to_c3per",
        "c4per_to_c3nfx",
        "c4per_to_c3ann",
        "c4per",
        "c4f10_SH",
        "c4f10_NH",
        "c4f10_GM",
        "c4ann_to_urban",
        "c4ann_to_secdn",
        "c4ann_to_secdf",
        "c4ann_to_range",
        "c4ann_to_pastr",
        "c4ann_to_c4per",
        "c4ann_to_c3per",
        "c4ann_to_c3nfx",
        "c4ann_to_c3ann",
        "c4ann",
        "c3per_to_urban",
        "c3per_to_secdn",
        "c3per_to_secdf",
        "c3per_to_range",
        "c3per_to_pastr",
        "c3per_to_c4per",
        "c3per_to_c4ann",
        "c3per_to_c3nfx",
        "c3per_to_c3ann",
        "c3per",
        "c3nfx_to_urban",
        "c3nfx_to_secdn",
        "c3nfx_to_secdf",
        "c3nfx_to_range",
        "c3nfx_to_pastr",
        "c3nfx_to_c4per",
        "c3nfx_to_c4ann",
        "c3nfx_to_c3per",
        "c3nfx_to_c3ann",
        "c3nfx",
        "c3f8_SH",
        "c3f8_NH",
        "c3f8_GM",
        "c3ann_to_urban",
        "c3ann_to_secdn",
        "c3ann_to_secdf",
        "c3ann_to_range",
        "c3ann_to_pastr",
        "c3ann_to_c4per",
        "c3ann_to_c4ann",
        "c3ann_to_c3per",
        "c3ann_to_c3nfx",
        "c3ann",
        "c2f6_SH",
        "c2f6_NH",
        "c2f6_GM",
        "bounds_time",
        "bounds_sector",
        "bounds_latitude",
        "bounds_altitude",
        "beta_b",
        "beta_a",
        "asy550",
        "asl",
        "areacello",
        "areacellg",
        "areacella",
        "ap",
        "aod_spmx",
        "aod_fmbg",
        "ann_cycle",
        "angstrom",
        "amv_index",
        "altitude",
        "added_tree_cover",
        "acabf",
        "WST",
        "VOC_openburning_share",
        "VOC_em_openburning",
        "VOC_em_anthro",
        "VOC_em_AIR_anthro",
        "VOC25_other_voc_em_speciated_VOC_anthro",
        "VOC25_other_voc_em_speciated_VOC",
        "VOC25-other_voc_em_speciated_VOC",
        "VOC24_acids_em_speciated_VOC_anthro",
        "VOC24_acids_em_speciated_VOC",
        "VOC24-acids_em_speciated_VOC",
        "VOC23_ketones_em_speciated_VOC_anthro",
        "VOC23_ketones_em_speciated_VOC",
        "VOC23-ketones_em_speciated_VOC",
        "VOC22_other_alka_em_speciated_VOC_anthro",
        "VOC22_other_alka_em_speciated_VOC",
        "VOC22-other_alka_em_speciated_VOC",
        "VOC21_methanal_em_speciated_VOC_anthro",
        "VOC21_methanal_em_speciated_VOC",
        "VOC21-methanal_em_speciated_VOC",
        "VOC20_chlorinate_em_speciated_VOC_anthro",
        "VOC20_chlorinate_em_speciated_VOC",
        "VOC20-chlorinate_em_speciated_VOC",
        "VOC19_ethers_em_speciated_VOC_anthro",
        "VOC19_ethers_em_speciated_VOC",
        "VOC19-ethers_em_speciated_VOC",
        "VOC18_esters_em_speciated_VOC_anthro",
        "VOC18_esters_em_speciated_VOC",
        "VOC18-esters_em_speciated_VOC",
        "VOC17_other_arom_em_speciated_VOC_anthro",
        "VOC17_other_arom_em_speciated_VOC",
        "VOC17-other_arom_em_speciated_VOC",
        "VOC16_trimethylb_em_speciated_VOC_anthro",
        "VOC16_trimethylb_em_speciated_VOC",
        "VOC16-trimethylb_em_speciated_VOC",
        "VOC15_xylene_em_speciated_VOC_anthro",
        "VOC15_xylene_em_speciated_VOC",
        "VOC15-xylene_em_speciated_VOC",
        "VOC14_toluene_em_speciated_VOC_anthro",
        "VOC14_toluene_em_speciated_VOC",
        "VOC14-toluene_em_speciated_VOC",
        "VOC13_benzene_em_speciated_VOC_anthro",
        "VOC13_benzene_em_speciated_VOC",
        "VOC13-benzene_em_speciated_VOC",
        "VOC12_other_alke_em_speciated_VOC_anthro",
        "VOC12_other_alke_em_speciated_VOC",
        "VOC12-other_alke_em_speciated_VOC",
        "VOC09_ethyne_em_speciated_VOC_anthro",
        "VOC09_ethyne_em_speciated_VOC",
        "VOC09-ethyne_em_speciated_VOC",
        "VOC08_propene_em_speciated_VOC_anthro",
        "VOC08_propene_em_speciated_VOC",
        "VOC08-propene_em_speciated_VOC",
        "VOC07_ethene_em_speciated_VOC_anthro",
        "VOC07_ethene_em_speciated_VOC",
        "VOC07-ethene_em_speciated_VOC",
        "VOC06_hexanes_pl_em_speciated_VOC_anthro",
        "VOC06_hexanes_pl_em_speciated_VOC",
        "VOC06-hexanes_pl_em_speciated_VOC",
        "VOC05_pentanes_em_speciated_VOC_anthro",
        "VOC05_pentanes_em_speciated_VOC",
        "VOC05-pentanes_em_speciated_VOC",
        "VOC04_butanes_em_speciated_VOC_anthro",
        "VOC04_butanes_em_speciated_VOC",
        "VOC04-butanes_em_speciated_VOC",
        "VOC03_propane_em_speciated_VOC_anthro",
        "VOC03_propane_em_speciated_VOC",
        "VOC03-propane_em_speciated_VOC",
        "VOC02_ethane_em_speciated_VOC_anthro",
        "VOC02_ethane_em_speciated_VOC",
        "VOC02-ethane_em_speciated_VOC",
        "VOC01_alcohols_em_speciated_VOC_anthro",
        "VOC01_alcohols_em_speciated_VOC",
        "VOC01-alcohols_em_speciated_VOC",
        "Toluene_lump",
        "TRA",
        "SO2_openburning_share",
        "SO2_em_openburning",
        "SO2_em_anthro",
        "SO2_em_SOLID_BIOFUEL_anthro",
        "SO2_em_AIR_anthro",
        "SO2",
        "SLV",
        "SHP",
        "RSLossRem",
        "RCO",
        "OC_openburning_share",
        "OC_em_openburning",
        "OC_em_anthro",
        "OC_em_SOLID_BIOFUEL_anthro",
        "OC_em_AIR_anthro",
        "OC",
        "NOx_openburning_share",
        "NOx_em_openburning",
        "NOx_em_anthro",
        "NOx_em_SOLID_BIOFUEL_anthro",
        "NOx_em_AIR_anthro",
        "NOx",
        "NMVOC_openburning_share",
        "NMVOC_em_openburning",
        "NMVOC_em_anthro",
        "NMVOC_em_SOLID_BIOFUEL_anthro",
        "NMVOC_em_AIR_anthro",
        "NMVOC_Toluene_lump_speciated_VOC_openburning_share",
        "NMVOC_Toluene_lump_em_speciated_VOC_openburning",
        "NMVOC_MEK_speciated_VOC_openburning_share",
        "NMVOC_MEK_em_speciated_VOC_openburning",
        "NMVOC_Higher_Alkenes_speciated_VOC_openburning_share",
        "NMVOC_Higher_Alkenes_em_speciated_VOC_openburning",
        "NMVOC_Higher_Alkanes_speciated_VOC_openburning_share",
        "NMVOC_Higher_Alkanes_em_speciated_VOC_openburning",
        "NMVOC_HOCH2CHO_speciated_VOC_openburning_share",
        "NMVOC_HOCH2CHO_em_speciated_VOC_openburning",
        "NMVOC_HCOOH_speciated_VOC_openburning_share",
        "NMVOC_HCOOH_em_speciated_VOC_openburning",
        "NMVOC_HCN_speciated_VOC_openburning_share",
        "NMVOC_HCN_em_speciated_VOC_openburning",
        "NMVOC_CH3OH_speciated_VOC_openburning_share",
        "NMVOC_CH3OH_em_speciated_VOC_openburning",
        "NMVOC_CH3COOH_speciated_VOC_openburning_share",
        "NMVOC_CH3COOH_em_speciated_VOC_openburning",
        "NMVOC_CH3COCHO_speciated_VOC_openburning_share",
        "NMVOC_CH3COCHO_em_speciated_VOC_openburning",
        "NMVOC_CH2O_speciated_VOC_openburning_share",
        "NMVOC_CH2O_em_speciated_VOC_openburning",
        "NMVOC_C8H10_speciated_VOC_openburning_share",
        "NMVOC_C8H10_em_speciated_VOC_openburning",
        "NMVOC_C7H8_speciated_VOC_openburning_share",
        "NMVOC_C7H8_em_speciated_VOC_openburning",
        "NMVOC_C6H6_speciated_VOC_openburning_share",
        "NMVOC_C6H6_em_speciated_VOC_openburning",
        "NMVOC_C5H8_speciated_VOC_openburning_share",
        "NMVOC_C5H8_em_speciated_VOC_openburning",
        "NMVOC_C3H8_speciated_VOC_openburning_share",
        "NMVOC_C3H8_em_speciated_VOC_openburning",
        "NMVOC_C3H6_speciated_VOC_openburning_share",
        "NMVOC_C3H6_em_speciated_VOC_openburning",
        "NMVOC_C3H6O_speciated_VOC_openburning_share",
        "NMVOC_C3H6O_em_speciated_VOC_openburning",
        "NMVOC_C2H6_speciated_VOC_openburning_share",
        "NMVOC_C2H6_em_speciated_VOC_openburning",
        "NMVOC_C2H6S_speciated_VOC_openburning_share",
        "NMVOC_C2H6S_em_speciated_VOC_openburning",
        "NMVOC_C2H5OH_speciated_VOC_openburning_share",
        "NMVOC_C2H5OH_em_speciated_VOC_openburning",
        "NMVOC_C2H4_speciated_VOC_openburning_share",
        "NMVOC_C2H4_em_speciated_VOC_openburning",
        "NMVOC_C2H4O_speciated_VOC_openburning_share",
        "NMVOC_C2H4O_em_speciated_VOC_openburning",
        "NMVOC_C2H2_speciated_VOC_openburning_share",
        "NMVOC_C2H2_em_speciated_VOC_openburning",
        "NMVOC_C10H16_speciated_VOC_openburning_share",
        "NMVOC_C10H16_em_speciated_VOC_openburning",
        "NMVOC",
        "NH3_openburning_share",
        "NH3_em_openburning",
        "NH3_em_anthro",
        "NH3_em_SOLID_BIOFUEL_anthro",
        "NH3_em_AIR_anthro",
        "NH3",
        "N2O",
        "MEK",
        "IND",
        "Higher_Alkenes",
        "Higher_Alkanes",
        "HOCH2CHO",
        "HCOOH",
        "HCN",
        "H2_openburning_share",
        "H2_em_openburning",
        "H2SO4_mass",
        "H2",
        "ENE",
        "Delta14co2_in_air",
        "CO_openburning_share",
        "CO_em_openburning",
        "CO_em_anthro",
        "CO_em_SOLID_BIOFUEL_anthro",
        "CO_em_AIR_anthro",
        "CO2_em_anthro",
        "CO2_em_AIR_anthro",
        "CO2",
        "CO",
        "CH4_openburning_share",
        "CH4_em_openburning",
        "CH4_em_anthro",
        "CH4_em_SOLID_BIOFUEL_anthro",
        "CH4_em_AIR_anthro",
        "CH4",
        "CH3OH",
        "CH3COOH",
        "CH3COCHO",
        "CH2O",
        "C8H10",
        "C7H8",
        "C6H6",
        "C5H8",
        "C3H8",
        "C3H6O",
        "C3H6",
        "C2H6S",
        "C2H6",
        "C2H5OH",
        "C2H4O",
        "C2H4",
        "C2H2",
        "C10H16",
        "BC_openburning_share",
        "BC_em_openburning",
        "BC_em_anthro",
        "BC_em_SOLID_BIOFUEL_anthro",
        "BC_em_AIR_anthro",
        "BC",
        "AIR",
        "AGR",
    ],
}

SUPPORTED_EXPERIMENTS = [
    "ssp585",
    "ssp370-lowNTCF",
    "ssp370",
    "ssp245",
    "ssp126",
    "piControl",
    "piClim-spAer-anthro",
    "piClim-spAer-aer",
    "piClim-lu",
    "piClim-histnat",
    "piClim-histghg",
    "piClim-histall",
    "piClim-histaer",
    "piClim-ghg",
    "piClim-control",
    "piClim-anthro",
    "piClim-aer",
    "piClim-N2O",
    "piClim-CH4",
    "piClim-4xCO2",
    "piClim-2xss",
    "piClim-2xdust",
    "piClim-2xVOC",
    "piClim-2xDMS",
    "pdSST-piArcSIC",
    "pdSST-pdSIC",
    "pdSST-futArcSIC",
    "midHolocene",
    "lig127k",
    "historical",
    "histSST-piNTCF",
    "histSST-piAer",
    "histSST",
    "hist-spAer-all",
    "hist-piNTCF",
    "hist-piAer",
    "hist-nat",
    "hist-aer",
    "hist-GHG",
    "amip",
]
# filepath to var to res Mapping
VAR_RES_MAPPING_PATH = "/home/charlie/Documents/MILA/causalpaca/data/data_description/mappings/variableid2tableid.csv"


GRIDDING_HIERACHY = ["gn"]

# skip subhr because only diagnostics for specific places
REMOVE_RESOLUTONS = [
    "suhbr"
]  # resolution endings to remove e.g. kick CFsubhr if this contains 'subhr'


RES_TO_CHUNKSIZE = {"year": 1, "mon": 12, "6hr": 1460, "3hr": 2920, "day":364}
