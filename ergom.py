from fabmcg.ingredients import (
    DiagnosticVariable,
    Model,
    InteriorStateVariable,
    BottomStateVariable,
    FloatParameter,
    BoolParameter,
    STANDARD_VARIABLES,
    StandardDependency
)
import numpy as np

secs_pr_day = 86400


class ERGOM(Model):
    # fmt: off
    sfl_po = FloatParameter("surface phosphate flux", "mmol P/m2/d", default=0.0015, scale_factor=1.0/secs_pr_day)
    sfl_ni = FloatParameter("surface nitrate flux", "mmol N/m2/d", default=0.09, scale_factor=1.0/secs_pr_day)
    sfl_am = FloatParameter("surface ammonium flux", "mmol N/m2/d", default=0.07, scale_factor=1.0/secs_pr_day)
    fluff = BoolParameter("use fluff", default=False)
    p10 = FloatParameter("background concentration of diatoms", "mmol N/m3", default=0.0225)
    p20 = FloatParameter("background concentration of flagellates", "mmol N/m3", default=0.0225)
    p30 = FloatParameter("background concentration of cyanobacteria", "mmol N/m3", default=0.0225)
    zo0 = FloatParameter("background concentration of zooplankton", "mmol N/m3", default=0.0225)
    kc = FloatParameter("specific light attenuation of phytoplankton and detritus", "m2/mmol N", default=0.03)
    i_min = FloatParameter("minimum optimal photosynthetically active radiation", "W/m2", default=25.0)
    r1max = FloatParameter("maximum growth rate of diatoms", "1/d", default=2.0, scale_factor=1.0/secs_pr_day)
    r2max = FloatParameter("maximum growth rate of flagellates", "1/d", default=0.7, scale_factor=1.0/secs_pr_day)
    r3max = FloatParameter("maximum growth rate of cyanobacteria", "1/d", default=0.5, scale_factor=1.0/secs_pr_day)
    alpha1 = FloatParameter("half-saturation nutrient concentration for diatoms", "mmol N/m3", default=1.35)
    alpha2 = FloatParameter("half-saturation nutrient concentration for flagellates", "mmol N/m3", default=0.675)
    alpha3 = FloatParameter("half-saturation nutrient concentration for cyanobacteria", "mmol N/m3", default=0.5)
    lpa = FloatParameter("phytoplankton exudation", "1/d", default=0.01, scale_factor=1.0/secs_pr_day)
    lpd = FloatParameter("phytoplankton mortality", "1/d", default=0.02, scale_factor=1.0/secs_pr_day)
    tf = FloatParameter("half-saturation temperature for flagellates", "deg C", default=10.0)
    tbg = FloatParameter("reference temperature for cyanobacteria", "deg C", default=14.0)
    beta_bg = FloatParameter("temperature sensitivity for cyanobacteria", "1/deg C", default=1.0)
    g1max = FloatParameter("maximum grazing rate on diatoms", "1/d", default=0.5, scale_factor=1.0/secs_pr_day)
    g2max = FloatParameter("maximum grazing rate on flagellates", "1/d", default=0.5, scale_factor=1.0/secs_pr_day)
    g3max = FloatParameter("maximum grazing rate on cyanobacteria", "1/d", default=0.25, scale_factor=1.0/secs_pr_day)
    lza = FloatParameter("quadratic zooplankton loss to dissolved matter", "m3/mmol N/d", default=0.0666666666, scale_factor=1.0/secs_pr_day)
    lzd = FloatParameter("quadratic zooplankton loss to detritus", "m3/mmol N/d", default=0.1333333333, scale_factor=1.0/secs_pr_day)
    iv = FloatParameter("Ivlev constant for grazing", "m3/mmol N", default=0.24444444)
    topt = FloatParameter("optimum temperature for grazing", "deg C", default=20.0)
    lan = FloatParameter("maximum nitrification rate", "1/d", default=0.1, scale_factor=1.0/secs_pr_day)
    oan = FloatParameter("half-saturation oxygen concentration for nitrification", "mmol o2/m3", default=0.01)
    beta_an = FloatParameter("temperature sensitivity of nitrification", "1/deg C", default=0.11)
    lda = FloatParameter("remineralisation rate", "1/d", default=0.003, scale_factor=1.0/secs_pr_day)
    tda = FloatParameter("half-saturation temperature for remineralisation", "deg C", default=13.0)
    beta_da = FloatParameter("temperature sensitivity of remineralisation", "-", default=20.0)
    pvel = FloatParameter("piston velocity", "m/d", default=5.0, scale_factor=1.0/secs_pr_day)
    sr = FloatParameter("phosphorus to nitrogen ratio", "mol P/mol N", default=0.0625)
    s1 = FloatParameter("reduced nitrate : oxidized detritus for denitrification", "-", default=5.3)
    s2 = FloatParameter("oxygen : nitrogen ratio for ammonium-based biomass synthesis and remineralisation", "mol O2/mol N", default=6.625)
    s3 = FloatParameter("oxygen : nitrogen ratio for nitrate-based biomass synthesis", "mol O2/mol N", default=8.625)
    s4 = FloatParameter("oxygen : nitrogen ratio for nitrification", "mol O2/mol N", default=2.0)
    a0 = FloatParameter("mmol O2 per gram", "mmol O2/g", default=31.25)
    a1 = FloatParameter("saturation mass concentration of oxygen at 0 deg C", "g/m3", default=14.603)
    a2 = FloatParameter("decrease in saturation mass concentration of oxygen per deg C", "g/m3/deg C", default=0.4025)
    lds = FloatParameter("sedimentation rate of detritus", "m/d", default=3.5, scale_factor=1.0/secs_pr_day)
    lsd = FloatParameter("resuspension rate of fluff", "1/d", default=25.0, scale_factor=1.0/secs_pr_day)
    tau_crit = FloatParameter("critical bottom stress", "N/m2", default=0.07)
    lsa = FloatParameter("fluff mineralisation rate", "1/d", default=0.001, scale_factor=1.0/secs_pr_day)
    bsa = FloatParameter("temperature sensitivity of fluff mineralisation", "1/deg C", default=0.15)
    ph1 = FloatParameter("inhibition of phosphate release during fluff mineralisation", "-", default=0.15)
    ph2 = FloatParameter("half-saturation oxygen concentration for inhibition of phosphate release", "mmol O2/m3", default=0.1)
    w_p1 = FloatParameter("vertical velocity of diatoms (< for sinking)", "m/d", default=-1.0, scale_factor=1.0/secs_pr_day)
    w_p2 = FloatParameter("vertical velocity of flagellates (< for sinking)", "m/d", default=-5.0, scale_factor=1.0/secs_pr_day)
    w_p3 = FloatParameter("vertical velocity of cyanobacteria (< for sinking)", "m/d", default=-5.0, scale_factor=1.0/secs_pr_day)
    w_de = FloatParameter("vertical velocity of detritus (< for sinking)", "m/d", default=-3.0, scale_factor=1.0/secs_pr_day)

    p1 = InteriorStateVariable("diatoms", "mmol N m-3", initial_value=4.5)
    p2 = InteriorStateVariable("flagellates", "mmol N m-3", initial_value=4.5)
    p3 = InteriorStateVariable("cyanobacteria", "mmol N m-3", initial_value=4.5)
    zo = InteriorStateVariable("zooplankton", "mmol N m-3", initial_value=4.5)
    de = InteriorStateVariable("detritus", "mmol N m-3", initial_value=4.5)
    o2 = InteriorStateVariable("oxygen", "mmol O2 m-3", initial_value=4.5)
    am = InteriorStateVariable("ammonium", "mmol N m-3", initial_value=4.5)
    ni = InteriorStateVariable("nitrate", "mmol N m-3", initial_value=4.5)
    po = InteriorStateVariable("phosphate", "mmol P m-3", initial_value=4.5)
    fl = BottomStateVariable("fluff", "mmol N m-2", initial_value=0.0)

    PARd = DiagnosticVariable("photosynthetically active radiation", "W/m2")
    GPP = DiagnosticVariable("gross primary production", "mmol/m3")
    NCP = DiagnosticVariable("net community production", "mmol/m3")
    PPR = DiagnosticVariable("gross primary production rate", "mmol/m3/d")
    NPR = DiagnosticVariable("net community production rate", "mmol/m3/d")

    I_0 = StandardDependency(STANDARD_VARIABLES.surface_downwelling_photosynthetic_radiative_flux)
    par = StandardDependency(STANDARD_VARIABLES.downwelling_photosynthetic_radiative_flux)
    temp = StandardDependency(STANDARD_VARIABLES.temperature)
    salt = StandardDependency(STANDARD_VARIABLES.practical_salinity)
    wind = StandardDependency(STANDARD_VARIABLES.wind_speed)
    taub = StandardDependency(STANDARD_VARIABLES.bottom_stress)
    # fmt: on

    def process_interior(self):
        wo = 30.0
        wn = 0.1

        # Light acclimation formulation based on surface light intensity
        iopt = max(0.25 * self.I_0, self.i_min)
        ppi = self.par / iopt * np.exp(1.0 - self.par / iopt)

        thopnp = th(self.o2, wo, 0.0, 1.0) * yy(wn, self.ni)
        thomnp = th(-self.o2, wo, 0.0, 1.0) * yy(wn, self.ni)
        thomnm = th(-self.o2, wo, 0.0, 1.0) * (1.0 - yy(wn, self.ni))
        thsum = thopnp + thomnp + thomnm
        thopnp = thopnp / thsum
        thomnp = thomnp / thsum
        thomnm = thomnm / thsum

        psum = self.p1 + self.p2 + self.p3 + self.p10 + self.p20 + self.p30

        llda = self.lda * (1.0 + self.beta_da * yy(self.tda, self.temp))
        llan = (
            th(self.o2, 0.0, 0.0, 1.0)
            * self.o2
            / (self.oan + self.o2)
            * self.lan
            * np.exp(self.beta_an * self.temp)
        )

        lp = self.lpa + self.lpd

        r1 = (
            self.r1max
            * min(
                yy(self.alpha1, self.am + self.ni),
                yy(self.sr * self.alpha1, self.po),
                ppi,
            )
            * (self.p1 + self.p10)
        )
        r2 = (
            self.r2max
            * (1.0 + yy(self.tf, self.temp))
            * min(
                yy(self.alpha2, self.am + self.ni),
                yy(self.sr * self.alpha2, self.po),
                ppi,
            )
            * (self.p2 + self.p20)
        )
        r3 = (
            self.r3max
            * 1.0
            / (1.0 + np.exp(self.beta_bg * (self.tbg - self.temp)))
            * min(yy(self.sr * self.alpha3, self.po), ppi)
            * (self.p3 + self.p30)
        )

        self.p1.source += (
            r1
            - fpz(self.iv, self.g1max, self.temp, self.topt, psum)
            * self.p1
            / psum
            * (self.zo + self.zo0)
            - lp * self.p1
        )
        self.p2.source += (
            r2
            - fpz(self.iv, self.g2max, self.temp, self.topt, psum)
            * self.p2
            / psum
            * (self.zo + self.zo0)
            - lp * self.p2
        )
        self.p3.source += (
            r3
            - fpz(self.iv, self.g3max, self.temp, self.topt, psum)
            * self.p3
            / psum
            * (self.zo + self.zo0)
            - lp * self.p3
        )
        self.zo.source += (
            (
                fpz(self.iv, self.g1max, self.temp, self.topt, psum) * self.p1
                + fpz(self.iv, self.g2max, self.temp, self.topt, psum) * self.p2
                + fpz(self.iv, self.g3max, self.temp, self.topt, psum) * self.p3
            )
            * (self.zo + self.zo0)
            / psum
            - self.lza * self.zo * (self.zo + self.zo0)
            - self.lzd * self.zo * (self.zo + self.zo0)
        )
        self.de.source += (
            self.lpd * (self.p1 + self.p2 + self.p3)
            + self.lzd * self.zo * (self.zo + self.zo0)
            - llda * self.de
        )
        self.am.source += (
            llda * self.de
            - llan * self.am
            - self.am / (self.am + self.ni) * (r1 + r2)
            + self.lpa * (self.p1 + self.p2 + self.p3)
            + self.lza * self.zo * (self.zo + self.zo0)
        )
        self.ni.source += (
            llan * self.am
            - self.ni / (self.am + self.ni) * (r1 + r2)
            - self.s1 * llda * self.de * thomnp
        )
        self.po.source += self.sr * (
            -r1
            - r2
            - r3
            + llda * self.de
            + self.lpa * (self.p1 + self.p2 + self.p3)
            + self.lza * self.zo * (self.zo + self.zo0)
        )
        self.o2.source += (
            self.s2 * (self.am / (self.am + self.ni) * (r1 + r2) + r3)
            - self.s4 * (llan * self.am)
            + self.s3 * (self.ni / (self.am + self.ni) * (r1 + r2))
            - self.s2 * (thopnp + thomnm) * llda * self.de
            - self.s2
            * (
                self.lpa * (self.p1 + self.p2 + self.p3)
                + self.lza * self.zo * (self.zo + self.zo0)
            )
        )

        # Set diagnostic variables
        self.PARd = self.par
        self.GPP = r1 + r2 + r3
        self.NCP = r1 + r2 + r3 - self.lpa * (self.p1 + self.p2 + self.p3)
        self.PPR = (r1 + r2 + r3) * secs_pr_day
        self.NPR = (
            r1 + r2 + r3 - self.lpa * (self.p1 + self.p2 + self.p3)
        ) * secs_pr_day

    def process_bottom(self):
        wo = 30.0
        wn = 0.1
        dot2 = 0.2

        thopnp = th(self.o2, wo, 0.0, 1.0) * yy(wn, self.ni)
        thomnp = th(-self.o2, wo, 0.0, 1.0) * yy(wn, self.ni)
        thomnm = th(-self.o2, wo, 0.0, 1.0) * (1.0 - yy(wn, self.ni))
        thsum = thopnp + thomnp + thomnm
        thopnp = thopnp / thsum
        thomnp = thomnp / thsum
        thomnm = thomnm / thsum

        llsa = self.lsa * np.exp(self.bsa * self.temp) * (th(self.o2, wo, dot2, 1.0))

        # Sedimentation dependent on bottom stress
        if self.tau_crit > self.taub:
            llds = self.lds * (self.tau_crit - self.taub) / self.tau_crit
        else:
            llds = 0.0

        # Resuspension dependent on bottom stress
        if self.tau_crit < self.taub:
            llsd = self.lsd * (self.taub - self.tau_crit) / self.tau_crit
        else:
            llsd = 0.0

        self.fl.source += (
            llds * self.de
            - llsd * self.fl
            - llsa * self.fl
            - th(self.o2, wo, 0.0, 1.0) * llsa * self.fl
        )
        self.de.bottom_flux += -llds * self.de + llsd * self.fl
        self.am.bottom_flux += llsa * self.fl
        self.ni.bottom_flux += -self.s1 * thomnp * llsa * self.fl
        self.po.bottom_flux += (
            self.sr
            * (1.0 - self.ph1 * th(self.o2, wo, 0.0, 1.0) * yy(self.ph2, self.o2))
            * llsa
            * self.fl
        )
        self.o2.bottom_flux += -(self.s4 + self.s2 * (thopnp + thomnm)) * llsa * self.fl

    def process_surface(self):
        newflux = 1
        if newflux == 1:
            sc = 1450.0 + (1.1 * self.temp - 71.0) * self.temp
            if self.wind > 13.0:
                p_vel = 5.9 * (5.9 * self.wind - 49.3) / np.sqrt(sc)
            elif self.wind < 3.6:
                p_vel = 1.003 * self.wind / (sc) ** (0.66)
            else:
                p_vel = 5.9 * (2.85 * self.wind - 9.65) / np.sqrt(sc)
            if p_vel < 0.05:
                p_vel = 0.05
            p_vel = p_vel / secs_pr_day
            flo2 = p_vel * (osat_weiss(self.temp, self.salt) - self.o2)
        else:
            flo2 = self.pvel * (self.a0 * (self.a1 - self.a2 * self.temp) - self.o2)
        self.o2.surface_flux += flo2

        self.ni.surface_flux += self.sfl_ni
        self.am.surface_flux += self.sfl_am
        self.po.surface_flux += self.sfl_po


def osat_weiss(t, s):
    aa1 = -173.4292
    aa2 = 249.6339
    a3 = 143.3483
    a4 = -21.8492
    b1 = -0.033096
    b2 = 0.014259
    b3 = -0.001700
    kelvin = 273.16
    mol_per_liter = 44.661

    tk = (t + kelvin) * 0.01
    return (
        np.exp(
            aa1 + aa2 / tk + a3 * np.log(tk) + a4 * tk + s * (b1 + (b2 + b3 * tk) * tk)
        )
        * mol_per_liter
    )


def th(x, w, min, max):
    if w > 1.0e-10:
        return min + (max - min) * 0.5 * (1.0 + np.tanh(x / w))
    elif x > 0.0:
        return 1.0
    else:
        return 0.0


def yy(a, x):
    return x**2 / (a**2 + x**2)


def fpz(iv, g, t, topt, psum):
    return (
        g
        * (1.0 + t**2 / topt**2 * np.exp(1.0 - 2.0 * t / topt))
        * (1.0 - np.exp(-(iv**2) * psum**2))
    )


if __name__ == "__main__":
    mod = ERGOM()
    mod.I_0 = 50
    mod.par = 10
    mod.temp = 15
    mod.salt = 35
    mod.wind = 5
    mod.taub = 0

    mod.describe()

    mod.process_interior()
    mod.process_bottom()
    mod.process_surface()

    print("Sources:")
    for v, s in zip(mod.state_variables.values(), mod.sources):
        print(f"  {v.name}: {v.long_name} = {s} {v.units} s-1")
    print("Bottom fluxes:")
    for v, s in zip(mod.interior_state_variables.values(), mod.bottom_fluxes):
        print(f"  {v.name}: {v.long_name} = {s} {v.units} m s-1")
    print("Surface fluxes:")
    for v, s in zip(mod.interior_state_variables.values(), mod.surface_fluxes):
        print(f"  {v.name}: {v.long_name} = {s} {v.units} m s-1")
