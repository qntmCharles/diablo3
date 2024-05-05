module parameters
  use fft
  use domain
  implicit none 
  save


  real(rkind)  time
  integer time_step, rk_step
  real(rkind) save_flow_time, save_stats_time, save_movie_time

  integer previous_time_step

  ! Input.dat
  character(len=35)   flavor
  logical             use_LES
  real(rkind)         nu
  real(rkind)         nu_start, nu_run
  real(rkind)         beta
  real(rkind)         delta_t, dt, delta_t_next_event, kick, ubulk0, px0

  logical             create_new_flow
  real(rkind)         wall_time_limit, time_limit
  real(rkind)         start_wall_time, previous_wall_time, end_wall_time

  logical             variable_dt, first_time
  logical             reset_time
  real(rkind)         CFL
  integer             update_dt, LES_start, time_nu_change

  real(rkind)         save_flow_dt, save_stats_dt
  real(rkind)         save_movie_dt

  logical             create_new_th(1:N_th)
  real(rkind)         Ri(1:N_th), Pr(1:N_th)
  logical             filter_th(1:N_th)
  integer             filter_int(1:N_th)

  integer     num_read_th
  integer     read_th_index(1:N_th)
  real(rkind) dTHdX(1:N_th), dTHdZ(1:N_th)
  real(rkind) dWdX ! Background vorticity


  integer     IC_type, f_type, turb_type
  logical     physical_noise
  logical     homogeneousX
  logical     check_flux

  ! Periodic
  real(rkind) ek0, ek, epsilon_target
  logical     background_grad(1:N_th)

  ! Rotating Flows
  real(rkind) Ro_inv, grav_x, grav_y, grav_z, delta

  ! Parameters for oscillatory forcing
  real(rkind) omega0, amp_omega0, force_start
  real(rkind) w_BC_Ymax_c1_transient

  ! Numerical parameters
  real(rkind)  h_bar(3), beta_bar(3), zeta_bar(3) ! For RK
  integer  time_ad_meth
  integer les_model_type

  ! Plume parameters
  real(rkind) R0, alpha_e, LYC, LYP, b0, F0, zvirt, N2, H, Tf, Tr, cent_x, cent_z
  real(rkind) b0_phi, F0_phi
  logical duration_flag
  integer jpert, test_rank

  ! Sponge parameters
  real (rkind) Svel_amp, Sb_amp, S_depth

  ! Forcing parameters
  real (rkind) tau_sponge

  ! Scatter plot parameters
  integer Nb, Nphi
  integer Nb_out, Nphi_out

  real(rkind) b_factor, phiv_factor, phic_factor, phip_factor
  real(rkind) phic_min, phic_max, phiv_min, phiv_max, b_min, b_max, db, dphic, dphiv, vd_zmin
  real(rkind) phip_min, phip_max, dphip

  real(rkind) source_vol, vol, flux_volume_v, flux_volume_c, flux_volume_p

  real(rkind), allocatable :: bbins(:), phicbins(:), phivbins(:), phipbins(:)
  real(rkind), allocatable :: bbins_out(:), phicbins_out(:), phivbins_out(:), phipbins_out(:)
  real(rkind), pointer, contiguous, dimension(:,:) :: weights, b_phiv_S, b_phiv_S_cum, b_phiv_S_mem, weights_vel
  real(rkind), pointer, contiguous, dimension(:,:) :: b_phic_S, b_phic_S_cum, b_phic_S_mem
  real(rkind), pointer, contiguous, dimension(:,:) :: b_phip_S, b_phip_S_cum, b_phip_S_mem
  real(rkind), pointer, contiguous, dimension(:,:) :: Ent_phiv_flux_mem, Ent_phiv_flux_cum, Ent_phiv_flux
  real(rkind), pointer, contiguous, dimension(:,:) :: Ent_phic_flux_mem, Ent_phic_flux_cum, Ent_phic_flux
  real(rkind), pointer, contiguous, dimension(:,:) :: Ent_phip_flux_mem, Ent_phip_flux_cum, Ent_phip_flux

  ! HDF5 writing
  real(rkind) DiagX(0:Nxp - 1)
  character(len=35) fname
  character(len=20) gname

  ! Moisture parameters
  real(rkind) alpha_m, beta_m, tau_m, q0, w_sediment, init_noise

  ! Shear parameters
  real(rkind) srate, smax_height, szero_height
  integer shear_type

  ! Mixing PDFs
  logical write_bins_flag

contains

  !----*|--.---------.---------.---------.---------.---------.---------.-|-------|
  subroutine init_parameters
    !----*|--.---------.---------.---------.---------.---------.---------.-|-------|

    real version, current_version

    integer i, j, k, n
    real(rkind) Re
    logical start_file_exists

    open (11, file='input.dat', form='formatted', status='old')

    ! Read input file.
    !   (Note - if you change the following section of code, update the
    !    CURRENT_VERSION number to make obsolete previous input files !)

    current_version = 3.12
    read (11, *)
    read (11, *)
    read (11, *)
    read (11, *)
    read (11, *) flavor, version
    if (version /= current_version) stop 'Wrong input data format.'
    read (11, *)
    read (11, *) use_mpi, use_LES, check_flux
    if (use_mpi .eqv. .false.) stop 'Serial processing has been deprecated in diablo3.'
    read (11, *)
    read (11, *) Re, beta, Lx, Lz, Ly
    nu_run = 1.d0 / Re
    read (11, *)
    read (11, *) nu_start, time_nu_change
    nu = nu_start ! initial nu, before changing to nu at time_nu_change
    read (11, *)
    read (11, *) num_per_dir, create_new_flow, LES_start
    read (11, *)
    read (11, *) wall_time_limit, time_limit, delta_t, reset_time, &
      variable_dt, CFL, update_dt
    read (11, *)
    read (11, *) verbosity, save_flow_dt, save_stats_dt, save_movie_dt, XcMovie, ZcMovie, YcMovie
    read (11, *)
    ! Read in the parameters for the N_th scalars
    do n = 1, N_th
      read (11, *)
      read (11, *) create_new_th(n)
      read (11, *)
      read (11, *) filter_th(n), filter_int(n)
      read (11, *)
      read (11, *) Ri(n), Pr(n)
    end do

    ! Initialize MPI Variables
    call init_mpi


    if (rank == 0) then
      write (*, *)
      write (*, *) '             ****** WELCOME TO DIABLO ******'
      write (*, *)
    end if

    inquire (file="start.h5", exist=start_file_exists)
    if (start_file_exists) then
      create_new_flow = .false.
      do n = 1, N_th
        create_new_th(n) = .false.
      end do
    end if


    ! Initialize case-specific packages
    if (num_per_dir == 3) then
      stop 'Error: Triply-Periodic Box has been deprecated!'
    elseif (num_per_dir == 2) then
      call input_chan
      call create_grid_chan
      call init_chan_mpi
      YcMovie = vd_zmin
      if (save_movie_dt /= 0) then
        call init_chan_movie
      end if
    elseif (num_per_dir == 1) then
      stop 'Error: Duct not implemented!'
    elseif (num_per_dir == 0) then
      stop 'Error: Cavity not implemented!'
    end if

    ! Now plume variables read, set perturbation level
    ! Find rank
    test_rank = -1
    if ((Lyc+Lyp < gy(Nyp)) .and. (gy(0) < Lyc+Lyp)) then
      test_rank = rankY
      !write(*,*) test_rank
    end if

    if (rankY == test_rank) then
      jpert = 0
      do while (gy(jpert) < Lyc+Lyp)
        jpert = jpert + 1
      end do
    end if

    if (rank == 0) &
      write (*, '("Use LES: " L1)') use_LES

    ! Initialize grid
    if (rank == 0) then
      write (*, '("Flavor:  ", A35)') flavor
      write (*, '("Nx    =  ", I10)') Nx
      write (*, '("Ny    =  ", I10)') Nz
      write (*, '("Nz(p) =  ", I10)') Nyp
      do n = 1, N_th
        write (*, '("Scalar Number: ", I2)') n
        write (*, '("  Richardson number = ", ES12.5)') Ri(n)
        write (*, '("  Prandtl number    = ", ES12.5)') Pr(n)
      end do
      write (*, '("Nu   = ", ES12.5)') nu
      write (*, '("Beta = ", ES12.5)') beta
    end if

    ! Initialize RKW3 parameters.
    h_bar(1) = delta_t * (8.d0 / 15.d0)
    h_bar(2) = delta_t * (2.d0 / 15.d0)
    h_bar(3) = delta_t * (5.d0 / 15.d0)
    beta_bar(1) = 1.d0
    beta_bar(2) = 25.d0 / 8.d0
    beta_bar(3) = 9.d0 / 4.d0
    zeta_bar(1) = 0.d0
    zeta_bar(2) = -17.d0 / 8.d0
    zeta_bar(3) = -5.d0 / 4.d0


    return
  end




  !----*|--.---------.---------.---------.---------.---------.---------.-|-------|
  subroutine input_chan
    !----*|--.---------.---------.---------.---------.---------.---------.-|-------|

    real version, current_version
    integer i, j, k, n
    real(rkind) ro

    ! Read in input parameters specific for channel flow case
    open (11, file='input_chan.dat', form='formatted', status='old')
    ! Read input file.

    current_version = 3.12
    read (11, *)
    read (11, *)
    read (11, *)
    read (11, *)
    read (11, *) version
    if (version /= current_version) &
      stop 'Wrong input data format input_chan'
    read (11, *)
    read (11, *) time_ad_meth
    read (11, *)
    read (11, *) les_model_type
    read (11, *)
    read (11, *) IC_type, kick, physical_noise, init_noise
    read (11, *)
    read (11, *) ro
    Ro_inv = 1.d0 / ro
    read (11, *)
    read (11, *) delta
    read (11, *)
    !read (11, *) dWdX ! Background vorticity
    !read (11, *)
    read (11, *) grav_x, grav_z, grav_y
    read (11, *)
    read (11, *) f_type, ubulk0, px0, omega0, amp_omega0, force_start, turb_type, tau_sponge
    read (11, *)
    read (11, *) r0, alpha_e, b0, b0_phi, cent_x, cent_z
    read (11, *)
    read (11, *) Lyc, Lyp, H, N2
    read (11, *)
    read (11, *) Tf, Tr, duration_flag
    read (11, *)
    read (11, *) Svel_amp, Sb_amp, S_depth
    read (11, *)
    read (11, *) Nb, Nphi, b_factor, phiv_factor, phic_factor, phip_factor
    read (11, *)
    read (11, *) alpha_m, beta_m, tau_m, q0, phiv_min, phic_min, phip_min, w_sediment
    read (11, *)
    read (11, *) srate, smax_height, szero_height, shear_type
    read (11, *)
    read (11, *)
    read (11, *) u_BC_Ymin, u_BC_Ymin_c1
    read (11, *)
    read (11, *) w_BC_Ymin, w_BC_Ymin_c1
    read (11, *)
    read (11, *) v_BC_Ymin, v_BC_Ymin_c1
    read (11, *)
    read (11, *) u_BC_Ymax, u_BC_Ymax_c1
    read (11, *)
    read (11, *) w_BC_Ymax, w_BC_Ymax_c1
    read (11, *)
    read (11, *) v_BC_Ymax, v_BC_Ymax_c1
    read (11, *)
    ! Read in boundary conditions and background gradients for the N_th scalars
    do n = 1, N_th
      read (11, *)
      read (11, *) th_BC_Ymin(n), th_BC_Ymin_c1(n)
      read (11, *)
      read (11, *) th_BC_Ymax(n), th_BC_Ymax_c1(n)
    end do

    if (rank == 0) write (*, '("Ro Inverse = " ES26.18)') Ro_inv
    do n = 1, N_th
      if (rank == 0) write (*,*) 'dTHdX', dTHdX(n)
      if (rank == 0) write (*,*) 'dTHdZ', dTHdZ(n)
    end do

    if (duration_flag .and. (rank == 0)) write (*,*) "Forcing will end at ", Tf, " post-penetration."
    if ((.not. duration_flag) .and. (rank == 0)) write (*,*) "Forcing will end at ", Tf, " after simulation start."

    if (q0 > phiv_min * exp(alpha_m * beta_m * H)) then
        if (rank == 0) write(*,*) "WARNING: minimum saturation concentration is below minimum concentration."
    end if

    zvirt = -r0/(1.2d0 * alpha_e)
    F0 = (r0**2.d0) * b0
    F0_phi = (r0**2.d0) * b0_phi

    ! Set level of noise during initialisation
    !init_noise = 1.d-6

    ! Set up scatter plot arrays
    b_min = 0.d0
    b_max = b_factor * N2 * (LY - H)
    !phiv_min = 1.d-5
    phiv_max = phiv_factor * 5.d0*F0_phi / (3.d0*alpha_e) * ((0.9d0 * alpha_e * F0_phi)**(-1.d0/3.d0)) * &
                              ((H + 5.d0*r0/(6.d0*alpha_e))**(-5.d0/3.d0))
    !phic_min = 1.d-4
    phic_max = phic_factor * 5.d0*F0_phi / (3.d0*alpha_e) * ((0.9d0 * alpha_e * F0_phi)**(-1.d0/3.d0)) * &
                              ((H + 5.d0*r0/(6.d0*alpha_e))**(-5.d0/3.d0))

    phip_max = phip_factor * 5.d0*F0_phi / (3.d0*alpha_e) * ((0.9d0 * alpha_e * F0_phi)**(-1.d0/3.d0)) * &
                              ((H + 5.d0*r0/(6.d0*alpha_e))**(-5.d0/3.d0))

    db = (b_max - b_min) / Nb
    dphic = (phic_max - phic_min) / Nphi
    dphiv = (phiv_max - phiv_min) / Nphi
    dphip = (phip_max - phip_min) / Nphi

    vd_zmin = H - (F0**(0.25d0)) * (N2**(-0.375d0))
    write (*, *) "vd zmin", vd_zmin
    
    Nb_out = int(ceiling(real(Nb)/NprocZ) * NprocZ)
    Nphi_out = int(ceiling(real(Nphi)/NprocZ) * NprocZ)
   
    allocate (bbins(1:Nb))
    allocate (phivbins(1:Nphi))
    allocate (phicbins(1:Nphi))
    allocate (phipbins(1:Nphi))

    allocate (Ent_phiv_flux(1:Nb_out, 1:Nphi_out))
    allocate (Ent_phiv_flux_mem(1:Nb_out, 1:Nphi_out))
    allocate (Ent_phiv_flux_cum(1:Nb_out, 1:Nphi_out))
    allocate (Ent_phic_flux(1:Nb_out, 1:Nphi_out))
    allocate (Ent_phic_flux_mem(1:Nb_out, 1:Nphi_out))
    allocate (Ent_phic_flux_cum(1:Nb_out, 1:Nphi_out))
    allocate (Ent_phip_flux(1:Nb_out, 1:Nphi_out))
    allocate (Ent_phip_flux_mem(1:Nb_out, 1:Nphi_out))
    allocate (Ent_phip_flux_cum(1:Nb_out, 1:Nphi_out))

    allocate (bbins_out(1:Nb_out))
    allocate (phivbins_out(1:Nphi_out))
    allocate (phicbins_out(1:Nphi_out))
    allocate (phipbins_out(1:Nphi_out))

    allocate (weights(1:Nb, 1:Nphi))
    allocate (weights_vel(1:Nb, 1:Nphi))

    allocate (b_phiv_S(1:Nb, 1:Nphi))
    allocate (b_phiv_S_cum(1:Nb, 1:Nphi))
    allocate (b_phiv_S_mem(1:Nb, 1:Nphi))
    allocate (b_phic_S(1:Nb, 1:Nphi))
    allocate (b_phic_S_cum(1:Nb, 1:Nphi))
    allocate (b_phic_S_mem(1:Nb, 1:Nphi))
    allocate (b_phip_S(1:Nb, 1:Nphi))
    allocate (b_phip_S_cum(1:Nb, 1:Nphi))
    allocate (b_phip_S_mem(1:Nb, 1:Nphi))

    b_phiv_S_mem = 0.d0
    Ent_phiv_flux_mem = 0.d0
    b_phic_S_mem = 0.d0
    Ent_phic_flux_mem = 0.d0
    b_phip_S_mem = 0.d0
    Ent_phip_flux_mem = 0.d0

    do i = 1, Nb
      bbins(i) = b_min + (i-0.5d0)*db
      bbins_out(i) = b_min + (i-0.5d0)*db
    end do

    do i = Nb + 1, Nb_out
      bbins_out(i) = -1.d0
    end do

    do i = 1, Nphi
      phivbins(i) = phiv_min + (i-0.5d0)*dphiv
      phivbins_out(i) = phiv_min + (i-0.5d0)*dphiv
      phicbins(i) = phic_min + (i-0.5d0)*dphic
      phicbins_out(i) = phic_min + (i-0.5d0)*dphic
      phipbins(i) = phip_min + (i-0.5d0)*dphip
      phipbins_out(i) = phip_min + (i-0.5d0)*dphip
    end do

    do i = Nphi + 1, Nphi_out
      phivbins_out(i) = -1.d0
      phicbins_out(i) = -1.d0
      phipbins_out(i) = -1.d0
    end do

    ! Compensate no-slip BC in the GS flow direction due to dTHdx
    !   AND also define dTHdx & dTHdz
    if (IC_Type == 4 .or. IC_Type == 5) then ! Infinite Front
      if (w_BC_Ymin == 1) then
        w_BC_Ymin_c1 = w_BC_Ymin_c1 - 1.d0
      end if
      if (w_BC_Ymax == 1) then
        w_BC_Ymax_c1 = w_BC_Ymax_c1 - 1.d0
      end if
      dTHdX(1) = Ro_inv / delta
      dTHdZ(1) = 0.d0

    else if (IC_Type == 6 .or. IC_Type == 7 .or. IC_Type == 8) then ! Finite Front
      if (w_BC_Ymin == 1) then
        w_BC_Ymin_c1 = w_BC_Ymin_c1 - 2.d0 * delta / Lx
      end if
      if (w_BC_Ymax == 1) then
        w_BC_Ymax_c1 = w_BC_Ymax_c1 - 2.d0 * delta / Lx
      end if
      dTHdX(1) = 2.d0 / Lx * Ro_inv
      dTHdZ(1) = 0.d0

    end if

    w_BC_Ymax_c1_transient = w_BC_Ymax_c1 ! Mean of surface forcing (compensating for GS flow)


    ! Set the valid averaging directions depending on the IC
    if (IC_Type == 5 .or. IC_Type == 6 .or. IC_Type == 7 .or. IC_Type == 8) then
      homogeneousX = .false.
    else ! Infinite, homogeneous front (or other IC...)
      homogeneousX = .true. ! Assume the x-direction is a valid averaging dimension
    endif


    return
  end




end module parameters
