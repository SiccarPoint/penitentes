# this code follows Cathles et al., 2014, and whence Cathles et al., 2011,
# for its approach

# This iteration of this code attempts to allow the surface to be cooler than
# the melting pt, which is explicitly assumed by Cathles. We need to thus add
# reradiation, temp changes, and a sublimation rate which is dependent on T,
# not the radiative flux itself (i.e., adopt the original Hobley (in prep.)
# method, following Lebovsky).

import numpy as np
from pylab import plot, figure, show
import scipy.interpolate as interp

true_albedo = 0.55  # Jeff's bolometric albedo
ice_density = 494.  # check these data
L_subl = 51000./0.018  # J/mol / kg/mol,
# the change enthalpy of sublimation div by molecular weight
true_incoming_intensity = 50.  # 50 Wm**-2
#
sun_declination = 0.  # orbital locking
latitude = 0.  # equator for now
emissivity = 0.9  # Jeff gives 0.9, 0.97 is standard ice
stefan_boltz = 5.670373e-8
thermal_cond = 4.2  # W/m/K ...bit of a guess at low T
## NB - Jeff's best fit gives conductivity*specific heat = 10
# assuming the SHC is solid-ish (as it scales to mass, so porosity already in):
# thermal_cond *~ 0.05
# let's assume this!!
thermal_cond *= 0.05  ####
#thermal_cond = 0.01  # ...if shc ~1000 (18/0.018)
# a conservative end member approach would be to share the fraction equally...
#note both thermal_cond & shc * 0.05 makes their product 10.

mobile_pts = False  # always leave this set to True!!

temp_pts = np.array([73.,93.,113.,133.,173.])  # temperatures in K for shc
shc = np.array([12.2,15.,17.3,19.8,24.8])/0.018  #specific heat capacity in J/kg/K
shc *= 0.05  ####
diffusivities = thermal_cond/ice_density/shc

initial_T = 106.  # coldest nighttime Ts at the equator are 90; NEVER below 73.
# this is Jeff's mean T
coldest_diffusivity = np.interp(initial_T, temp_pts, diffusivities)

num_pts = 50  # 100
pt_spacing = 0.02  # 0.01
num_sinusoids = 1
amplitude = 1.*num_pts*pt_spacing/num_sinusoids  # per Cathles
array_length = num_pts*pt_spacing  # note we incorporate the final segment
# array_length is only the x dimension
tstep = 60.*60.*24.*365.*1.  # an earth year, in earth secs (1/3.55 Eudays)
tstep = 400.
num_tsteps = 1000  # 10000
num_tsteps = 800  # a little after midday
# initialize them as a flat surface
node_x = np.arange(num_pts, dtype=float)*pt_spacing
node_z = np.zeros(num_pts, dtype=float) + np.random.rand(num_pts)/10.
# node_z[::2] = -0.1
# Cathles uses a sinusoid...
node_z = amplitude/2.*np.sin(node_x/array_length*2.*np.pi*num_sinusoids)
node_z.fill(0.)

# this version of the code tries to model a single day on europa.
# our day will start at 6am, the coldest time
a_europa_day = 60.*60.*24.*3.55  # in secs
def get_hour_angle(time_elapsed):
    # the clock starts at dawn
    fraction_of_day = (time_elapsed/a_europa_day)%1.
    if 0. <= fraction_of_day < 0.5:
        hour_angle = np.arcsin(4.*fraction_of_day-1.)
    else:
        hour_angle = None  #nighttime
    return hour_angle

# node_z[2] -= 1
# ^flat, with noise
num_segments = num_pts

# save the init conds:
init_node_x = node_x.copy()
init_node_z = node_z.copy()

node_x_exp = np.empty(num_pts*3, dtype=float)
node_z_exp = np.empty(num_pts*3, dtype=float)
node_x_exp[num_pts:(2*num_pts)] = node_x
node_x_exp[:num_pts] = node_x - array_length
node_x_exp[(2*num_pts):] = node_x + array_length
node_z_exp[num_pts:(2*num_pts)] = node_z
node_z_exp[:num_pts] = node_z
node_z_exp[(2*num_pts):] = node_z
seg_dx = np.empty(num_pts*3-1, dtype=float)
seg_dz = np.empty_like(seg_dx)
seg_centx = np.empty_like(seg_dx)
seg_centz = np.empty_like(seg_dx)
seg_length = np.empty_like(seg_dx)
seg_angle = np.empty_like(seg_dx)
seg_m = np.empty_like(seg_dx)
seg_c = np.empty_like(seg_dx)
new_contour_cumlen = np.zeros(num_pts+1, dtype=float)
new_lengths = np.zeros(num_pts+1, dtype=float)
deepest_elev = np.empty(num_tsteps, dtype=float)  # this stores the pit depths
num_section_pts = 40
seg_section_T = np.empty((num_pts*3-1, num_section_pts), dtype=float)
# ^this is a section down into each seg, used for transient T calcs
seg_section_T.fill(initial_T)
seg_section_fluxes = np.zeros((num_pts*3-1, num_section_pts+1), dtype=float)
# ^this is the heat fluxes up and down the vertical section
surf_T = seg_section_T[:,0]
surf_T_record = []
E_record = []

def diffuse_heat(flux_in, seg_section_T, seg_section_fluxes, seg_length):
    """
    This little function attempts to implement transient 1D heat diffusion into
    the ice surface for a single segment.

    flux_in is a num_segments long array, and is the arriving
    absorbed radiative flux (W/m). It does NOT include the reradiated part.
    """
    deep_Ts = seg_section_T[:,-1].copy()
    node_spacing = 0.002  # 0.3 mm
    dt_int = 0.25 * node_spacing * node_spacing / coldest_diffusivity
    # ^conservative von Neumann condition
    repeats = int(tstep // dt_int) + 1
    dt_int = tstep/repeats  # again, conservative
    # now do the loop:
    for i in xrange(repeats):
        # get the radiative loss:
        fluxes_out = emissivity*stefan_boltz*seg_section_T[:,0]**4  #W/m**2
        # get the gradients
        grads = (seg_section_T[:,:-1] - seg_section_T[:,1:])/node_spacing
        # get the diffusivities
        link_mean_T = (seg_section_T[:,:-1] + seg_section_T[:,1:])/2.
        alphas = np.interp(link_mean_T, temp_pts, diffusivities)
        # calc the fluxes
        seg_section_fluxes[:,1:-1] = alphas*grads  # +ve OUT
        # do the BCs
        seg_section_fluxes[:,0] = (flux_in/seg_length-fluxes_out)
        ###seg_section_fluxes[:,-1] = seg_section_fluxes[:,-2]
        # (free boundary at depth)
        ###now fixing T at depth...
        seg_section_fluxes *= seg_length.reshape((seg_length.size,1))  # make them true fluxes
        seg_section_T[:,:] += (seg_section_fluxes[:,:-1] -
                               seg_section_fluxes[:,1:])*dt_int
        seg_section_T[:,-1] = deep_Ts  ###

    return seg_section_T[:,0]  # return the surface Ts


for t in xrange(num_tsteps):
    print(t)
    seg_dx[:] = node_x_exp[1:]-node_x_exp[:-1]
    seg_dz[:] = node_z_exp[1:]-node_z_exp[:-1]
    seg_centx[:] = (node_x_exp[1:]+node_x_exp[:-1])/2.
    seg_centz[:] = (node_z_exp[1:]+node_z_exp[:-1])/2.
    seg_length[:] = np.sqrt(np.square(seg_dx)+np.square(seg_dz))
    seg_angle[:] = np.arctan2(seg_dz, seg_dx)
    # consistent w Cathles, ccw from vertical is +ve; describes the normal
    seg_m[:] = seg_dz/seg_dx
    seg_c[:] = node_z_exp[:-1] - seg_m*node_x_exp[:-1]
    angle_factor = np.empty((seg_centx.size, seg_centx.size), dtype=float)
    connectivity_matrix = np.empty_like(angle_factor, dtype=bool)

    # hour_angle = 0.  # midday
    hour_angle = get_hour_angle(t*tstep)
    if hour_angle is not None:  #daytime
        true_sun_zenith_angle = np.arccos(np.sin(latitude)*np.sin(
                                    sun_declination) + np.cos(latitude)*np.cos(
                                    sun_declination)*np.cos(hour_angle))
        altitude_angle = np.pi/2. - true_sun_zenith_angle
        sin_az = np.sin(hour_angle)*np.cos(sun_declination)/np.cos(altitude_angle)
        true_sun_az_angle = np.arcsin(sin_az.clip(-1.,1.))
        # ...per wiki & itacanet.org page
        #print((np.sin(hour_angle),np.cos(sun_declination),np.cos(altitude_angle)))
        eff_zenith = np.arctan(np.tan(true_sun_zenith_angle)*np.cos(true_sun_az_angle))
        # ^this will break is true zenith == true az == np.pi/2 exactly
        eff_intensity_factor = np.sqrt(np.cos(true_sun_zenith_angle)**2 +
                                       (np.sin(true_sun_zenith_angle) *
                                        np.cos(true_sun_az_angle))**2)
        # ...assuming structures form E-W
    else:
        eff_zenith = 0.
        eff_intensity_factor = 0.  #no light allowed
        ## ^^this section now VERIFIED as well behaved!!

    if eff_intensity_factor > 0.:  # if it's night, this is all null
        # derive the sky windows
        # get angle between seg center and all other nodes:
        # ******Note this section ignores looping, i.e., shading from other end
        beta_L = np.zeros(num_pts*3, dtype=float)
        beta_R = np.zeros_like(beta_L)
        which_node_L = np.zeros(num_pts*3, dtype=int)
        which_node_R = np.zeros_like(which_node_L)
        poss_angles_L = np.zeros((num_pts*3, num_pts*3), dtype=float)
        poss_angles_R = np.zeros_like(poss_angles_L)
        for i in xrange(num_segments*3-1):
            poss_angles_L[i, :(i+1)] = (np.arctan2(node_z_exp[:(i+1)]-seg_centz[i],
                                        node_x_exp[:(i+1)]-seg_centx[i])-0.5*np.pi)
            poss_angles_R[i, (i+1):] = (np.arctan2(node_z_exp[(i+1):]-seg_centz[i],
                                        node_x_exp[(i+1):]-seg_centx[i])-0.5*np.pi)
            poss_angles_L[i, :(i+1)] = np.where(poss_angles_L[i, :(i+1)] <= -np.pi,
                                                poss_angles_L[i, :(i+1)]+2.*np.pi,
                                                poss_angles_L[i, :(i+1)])
            poss_angles_R[i, (i+1):] = np.where(poss_angles_R[i, (i+1):] <= -np.pi,
                                                poss_angles_R[i, (i+1):]+2.*np.pi,
                                                poss_angles_R[i, (i+1):])
            which_node_L[i] = np.argmin(poss_angles_L[i, :(i+1)])
            # min because they're +ve
            which_node_R[i] = np.argmax(poss_angles_R[i, (i+1):]) + i + 1
            # max because they're -ve
            # final additions to put this into "real" IDs
            beta_L[i] = poss_angles_L[i, :(i+2)].flatten()[which_node_L[i]]
            beta_R[i] = poss_angles_R[i, (i+1):].flatten()[which_node_R[i]-i-1]
        # assert np.all(np.less_equal(beta_R, 0.))
        # assert np.all(np.greater_equal(beta_L, 0.))
        # ...this actually isn't true in the general case... overhangs!
        # But think the above still holds

        # get illumination fraction
        # if the beta angle is the same as the seg_angle, then whole thing
        # is illuminated, or it's not (i.e., it's self shaded).
        # Interesting cases arise when a vertex not at the ends of the segment
        # can shade it
        shaded_L = np.greater(eff_zenith, beta_L[:-1])
        shaded_R = np.less(eff_zenith, beta_R[:-1])
        center_illum = np.logical_not(np.logical_or(shaded_R,
                                                    shaded_L)).astype(float)
        # ^this is the fraction we want
        # find the points where we change from full to no illum of centers:
        changed_illum_L = np.where(np.diff(shaded_L))[0] + 1
        changed_illum_R = np.where(np.diff(shaded_R))[0] + 1
        # ^there *can* be more than 1 ID in these arrays
        angle_next_node_notL = np.arctan2(-node_z_exp[changed_illum_L]+node_z_exp[
                                            which_node_L[changed_illum_L]],
                                          -node_x_exp[changed_illum_L]+node_x_exp[
                                            which_node_L[changed_illum_L]]
                                          ) - 0.5*np.pi
        # A negative val here indicates the node looked AT ITSELF
        # ...note this is actually the next node right, but relevant to the
        # "L" labelled variables...!
        # The ID for the segment left of the node is changed_illum_L-1
        # The ID for the segment right of the node is changed_illum_L
        for i in xrange(angle_next_node_notL.size):
            if (eff_zenith > angle_next_node_notL[i]) and (
                    angle_next_node_notL[i] >= 0):
                # the node is shadowed, and the segment TO ITS RIGHT is partially
                # illuminated (i.e., it's illum is DECREASED from 1.)
                center_illum[changed_illum_L[i]] -= 0.5*(1. - (beta_L[
                    changed_illum_L[i]] - eff_zenith)/(beta_L[changed_illum_L[i]] -
                                                       angle_next_node_notL[i]))
                # use minus not times as it's possible we're further decreased on
                # the same segment from the RHS
            elif eff_zenith < angle_next_node_notL[i]:
                # the node is illuminated, and the segment TO ITS LEFT is partially
                # illuminated (i.e., it's illum is INCREASED from 0.)
                center_illum[changed_illum_L[i]-1] += 0.5*(angle_next_node_notL[
                                                           i] - eff_zenith)/(
                                                           angle_next_node_notL[
                                                             i] - beta_L[
                                                             changed_illum_L[i]-1])
            else:  # self-shadowed seg or perfectly grazing light => no changes
                pass
        # repeat for the RHS
        xdiff = node_x_exp[which_node_R[changed_illum_R]] - node_x_exp[
                                                                changed_illum_R]
        angle_next_node_notR = np.arctan2(-node_z_exp[changed_illum_R] +
                                          node_z_exp[which_node_R[
                                              changed_illum_R]], xdiff) - 0.5*np.pi
        for i in xrange(angle_next_node_notR.size):
            # remember, angles are now ALL NEGATIVE
            if eff_zenith < angle_next_node_notR[i]:
                # node is shadowed, the segment to its left has its illum
                # decreased from 1.
                center_illum[
                    changed_illum_R[i]-1] -= 0.5*(eff_zenith -
                                                  angle_next_node_notR[i])/(
                                                  beta_R[changed_illum_R[i]-1] -
                                                  angle_next_node_notR[i])
                # -ves hopefully sort themselves out
            elif eff_zenith > angle_next_node_notR[i] and (
                    angle_next_node_notR[i] != beta_R[changed_illum_R[i]]):
                    # 2nd condition excl. self-shadowing
                center_illum[
                    changed_illum_R[i]] += 0.5*(angle_next_node_notR[i] -
                                                eff_zenith)/(
                                                angle_next_node_notR[i] -
                                                beta_R[changed_illum_R[i]])
            else:
                pass

        # calc direct illumination terms:
        part = center_illum * true_incoming_intensity * eff_intensity_factor *\
                    seg_length * np.cos(seg_angle-eff_zenith)
        # calc reradiation:
        rerad = seg_length * emissivity * stefan_boltz * surf_T**4
        R_d = (true_albedo*part) + rerad
        E_d = (1.-true_albedo)*part

        # get the angle factor
        cent_dists_to_all_nodes = np.sqrt(np.square(seg_centx.reshape((
                seg_centx.size, 1)) - node_x_exp.reshape((1, node_x_exp.size))) +
                np.square(seg_centz.reshape((seg_centz.size, 1)) -
                          node_z_exp.reshape((1, node_z_exp.size))))
        arccos_frag = (np.square(cent_dists_to_all_nodes[:, :-1]) +
                       np.square(cent_dists_to_all_nodes[:, 1:]) -
                       np.square(seg_length))/(2.*cent_dists_to_all_nodes[:, :-1] *
                                               cent_dists_to_all_nodes[:, 1:])
        arccos_frag[arccos_frag > 1.] = 1.
        arccos_frag[arccos_frag < -1.] = -1.
        # ...some kind of rounding error was getting in here.
        angle_factor[:, :] = np.arccos(arccos_frag)/np.pi
        # angle_factor[np.eye(angle_factor.shape[0], dtype=bool)] = 0.
        # ^this isn't necessary as we do it via the connectivity matrix

        # now the connectivity matrix
        # this is CRAZY slow, so only do it once every 20 steps! Form rarely
        # changes fast enough for this to matter
        # NB: the resampling scotches this. Might be possible to fudge it??
        if True:  # t % (num_tsteps//1) == 0:
            # do segments face each other?
            center_angles = (np.arctan2(seg_centz.reshape((1, seg_centz.size)) -
                                        seg_centz.reshape((seg_centz.size, 1)),
                                        seg_centx.reshape((1, seg_centx.size)) -
                                        seg_centx.reshape((seg_centx.size, 1))) -
                             0.5*np.pi)
            # note self gets -pi/2.
            center_angles = np.where(center_angles <= -np.pi, center_angles+2.*np.pi,
                                     center_angles)
            angle_between = center_angles - seg_angle.reshape((seg_angle.size, 1))
            connect_oneway = np.greater(np.cos(angle_between), 0.)
            # connect_oneway = np.logical_and(connect_oneway, connect_oneway.T)
            # ^ we do this below...
            connect_oneway[np.eye(connect_oneway.shape[0], dtype=bool)] = False
            ##figure(0)
            ##plot(node_x_exp[:-1], np.sum(connect_oneway, axis=0))
            # ^can't illuminate yourself. Parallel surfaces may or may not be true
            # now line of sight. We'll have to do this the crude way, I think
            # this is PAINFULLY SLOW
            for i in xrange(num_pts*3-2):
                for j in xrange(i+1, num_pts*3-1):
                    head_x = seg_centx[i]
                    head_z = seg_centz[i]
                    tail_x = seg_centx[j]
                    tail_z = seg_centz[j]
                    node_x_vals = node_x_exp[(i+1):(j+1)]
                    node_z_vals = node_z_exp[(i+1):(j+1)]
                    line_grad = (tail_z-head_z)/(tail_x-head_x)
                    line_const = head_z-line_grad*head_x
                    proj_z_vals = line_grad*node_x_vals + line_const
                    # ^the vals the nodes "would have" if on the line
                    if not np.logical_or(np.all(np.greater(proj_z_vals, node_z_vals)),
                                         np.all(np.greater(node_z_vals, proj_z_vals))):
                        # ...not all above, or all below
                        # set the connectivity to 0
                        connect_oneway[i, j] = False
                        # the transposition below takes care of [j,i]
            connectivity_matrix[:, :] = np.logical_and(
                                        connect_oneway, connect_oneway.T)
            # ^BOTH normals must point at each other

        # now, solve the matrix:
        A_ij = true_albedo*angle_factor.T*connectivity_matrix*seg_length
        # ...Cathles had dropped the connectivity_matrix is his equ. A13
        identity_less_A = np.identity(A_ij.shape[0]) - A_ij
        R = np.linalg.solve(identity_less_A, R_d)

        E = E_d + np.sum((1.-true_albedo) * seg_length *
                         connectivity_matrix*angle_factor.T * R,
                         axis=1)
    else:
        E = 0.  # it's nighttime, baby

    E_record.append(E)

    surf_T = diffuse_heat(E, seg_section_T, seg_section_fluxes, seg_length)
    surf_T_record.append(surf_T.mean())

    # now we're going to implement Lebofsky's method for sublimation rate:
    vaporP = 133.322368*np.power(10., -2445.5646/surf_T +
                                 8.2312*np.log10(surf_T) - 0.0167706*surf_T +
                                 1.20514e-5*np.square(surf_T) - 6.757169)
    # now Hertz-Knudsen:
    Hdot_perseg_timestime = vaporP * np.sqrt(0.0180154 /
                                             (np.pi*8.3144621*surf_T)) / \
                            ice_density * tstep
### does there need to be a division by seg_length here?

    #Hdot_perseg_timestime = E/(ice_density*L_subl*seg_length)*tstep

    A_ij2 = np.identity(seg_length.size, dtype=float) * seg_length/3.
    A_ij2[1:, 1:] += np.identity(seg_length.size-1,
                                 dtype=float) * seg_length[:-1]/3.
    A_ij2[0, 0] += seg_length[-1]/3.
    # the factor of 2 arises as each node appears on 2 segments
    # A_ij2[0, 0] = seg_length[0]/3.
    # A_ij2[-1, -1] = seg_length[-1]/3.  # invoke looped BCs
    A_ij2[0, 1] = seg_length[0]/6.
    A_ij2[0, -1] = seg_length[-1]/6.
    A_ij2[-1, -2] = seg_length[-1]/6.  # entries we'll miss otherwise
    A_ij2[-1, 0] = seg_length[-2]/6.
    for i in xrange(1, seg_length.size-1):
        A_ij2[i, i-1] = seg_length[i-1]/6.
        A_ij2[i, i+1] = seg_length[i]/6.
    bdot_hyp = Hdot_perseg_timestime*seg_length/2.
    bdot_x = bdot_hyp*np.sin(seg_angle)  # formerly -ve, but didn't make sense
    bdot_z = -bdot_hyp*np.cos(seg_angle)  # for each seg
    F_rhs_x = np.zeros(seg_length.size, dtype=float)  # for each node
    F_rhs_z = np.zeros(seg_length.size, dtype=float)  # for each node
    for (F, b) in zip((F_rhs_x, F_rhs_z), (bdot_x, bdot_z)):
        F[0] = b[0] + b[-1]  # invoking looped BCs
        for i in xrange(1, F.size):
            F[i] = b[i-1] + b[i]  # why is this looking back not forward?
    u_i = np.linalg.solve(A_ij2, F_rhs_x)
    w_i = np.linalg.solve(A_ij2, F_rhs_z)

    # disable these lines to test the resampling procedure...
    node_x_exp[:-1] += u_i
    node_z_exp[:-1] += w_i

    # ...Cathles then regrids.
    fn = interp.interp1d(node_x_exp, node_z_exp)
    # node_x = init_node_x.copy()
    # node_z = fn(node_x)

    if mobile_pts is True:
        # replace the above with mechanism to try to prevent bunching at tips:
        # hard part is the new spacing of node_x
        # get the total length:
        seg_dx_temp = node_x_exp[(num_pts+1):(2*num_pts+1)]-node_x_exp[
                                                           num_pts:(2*num_pts)]
        seg_dz_temp = node_z_exp[(num_pts+1):(2*num_pts+1)]-node_z_exp[
                                                           num_pts:(2*num_pts)]
        # force the first and last pts to be effectively immobile:
        # seg_dx_temp[0] = node_x_exp[num_pts+1]  # node_x[0] = 0 always
        seg_dx_temp[0] = node_x_exp[num_pts+1] - node_x_exp[num_pts]
        # seg_dx_temp[-1] = array_length-node_x_exp[2*num_pts-1]
        seg_dx_temp[-1] = node_x_exp[2*num_pts]-node_x_exp[2*num_pts-1]
        # seg_dz_temp[0] = node_z_exp[num_pts+1] - fn(0.)
        seg_dz_temp[0] = node_z_exp[num_pts+1] - node_z_exp[num_pts]
        # seg_dz_temp[-1] = fn(array_length)-node_z_exp[2*num_pts-1]
        seg_dz_temp[-1] = node_z_exp[2*num_pts]-node_z_exp[2*num_pts-1]
        seg_length_temp = np.sqrt(np.square(seg_dx_temp) +
                                  np.square(seg_dz_temp))
        new_contour_cumlen[1:] = np.cumsum(seg_length_temp)
        new_dl = new_contour_cumlen[-1]/num_pts
        new_lengths = np.arange(0, new_dl*(num_pts+1), new_dl)
        # rounding errors can appear here, so
        if new_lengths.size > num_pts+1:
            new_lengths = new_lengths[:-1]
            assert new_lengths.size == num_pts+1
        new_pos_byl = np.searchsorted(new_contour_cumlen, new_lengths[:-1])
        # note that new elements get inserted BEFORE equal values => prob w 0
        # so...
        atstart = np.equal(new_contour_cumlen[new_pos_byl],
                           new_lengths[new_pos_byl])
        atstart[-1] = False  # don't move the last one!
        new_pos_byl[atstart] += 1  # increment these into the next interval
        prop_along_lines = (new_lengths[:-1] -
                            new_contour_cumlen[new_pos_byl-1])/(
                            new_contour_cumlen[new_pos_byl] -
                            new_contour_cumlen[new_pos_byl-1])
        # now we can create the new node_x and node_z:
        node_x[:] = node_x_exp[num_pts+new_pos_byl-1] + prop_along_lines*(
            node_x_exp[num_pts+new_pos_byl] -
            node_x_exp[num_pts+new_pos_byl-1])

        # add some randomness
        # DOING THIS ADDS SIGNIFICANT DIFFUSION!!!
        offset = (np.random.rand(1.)-0.5)*array_length/num_pts
        node_x += offset

        # node_z = fn(node_x)
        # The problem with the diffusion is getting in because the resampled
        # pts are ALWAYS lower than the existing pts. The way to solve this is
        # to try a quadratic resampling?
        for i in xrange(node_x.size):
            fnparam = np.polyfit(node_x_exp[(num_pts+new_pos_byl[i]-2):
                                            (num_pts+new_pos_byl[i]+2)],
                                 node_z_exp[(num_pts+new_pos_byl[i]-2):
                                            (num_pts+new_pos_byl[i]+2)],
                                 2)
            # incorporates 1 seg on either side of the seg of interest (4 pts)
            fnquad = np.poly1d(fnparam)
            node_z[i] = fnquad(node_x[i])
    else:
        node_x[:] = node_x_exp[num_pts:(2*num_pts)]
        node_z[:] = node_z_exp[num_pts:(2*num_pts)]

    node_x_exp[num_pts:(2*num_pts)] = node_x
    node_x_exp[:num_pts] = node_x - array_length
    node_x_exp[(2*num_pts):] = node_x + array_length
    node_z_exp[num_pts:(2*num_pts)] = node_z
    node_z_exp[:num_pts] = node_z
    node_z_exp[(2*num_pts):] = node_z

    # # plot the new output
    # if t % (num_tsteps//20) == 0:
    #     figure(1)
    #     plot(node_x, node_z)
    # # check we're still running...
    # if t % 100 == 0:
    #     print 'time ', t

    deepest_elev[t] = node_z.min()

figure(1)
plot(init_node_x, init_node_z, 'k')

figure(2)
plot(node_x, node_z)

figure(3)
plot(init_node_x-init_node_x[np.argmin(init_node_z)], init_node_z-init_node_z.min(), 'k')
plot(node_x-node_x[np.argmin(node_z)], node_z-node_z.min(), 'r')

figure(4)
plot((np.arange(num_tsteps, dtype=float)+1.)*tstep, deepest_elev)

figure(5)
plot(node_x_exp, node_z_exp)
figure(6)
plot(node_x_exp[:-1], np.sum(connectivity_matrix, axis=0))

figure(7)
plot(node_x_exp[:-1], surf_T)

show()
