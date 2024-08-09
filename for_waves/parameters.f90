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
  real(rkind)         save_stats_dt_fine, fine_time
  real(rkind)         save_movie_dt
  real(rkind)         Tb, nb_period, base_time

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
  real(rkind) R0, alpha_e, LYC, LYP, b0, F0, zvirt, N2, H
  integer jpert, test_rank

  ! Sponge parameters
  real (rkind) Svel_amp, Sb_amp, S_depth, Svel_side_amp, S_side_depth

  ! Forcing parameters
  real (rkind) tau_sponge

  ! Scatter plot parameters
  integer Nb, Nphi
  integer Nb_out, Nphi_out

  real(rkind) b_factor, phi_factor
  real(rkind) phi_min, phi_max, b_min, b_max, db, dphi, vd_zmin
  integer ranky_vd_zmin, Ny_vd_zmin

  real(rkind) source_vol, vol, flux_volume

  real(rkind), allocatable :: bbins(:), phibins(:)
  real(rkind), allocatable :: bbins_out(:), phibins_out(:)
  real(rkind), pointer, contiguous, dimension(:,:) :: weights, weights_flux, weights_flux_cum, weights_flux_mem, weights_vel
  real(rkind), pointer, contiguous, dimension(:,:) :: Ent_phi_flux_mem, Ent_phi_flux_cum, Ent_phi_flux
  real(rkind), pointer, contiguous, dimension(:,:) :: boundary_F_mem, boundary_F_cum, boundary_F

  ! HDF5 writing
  real(rkind) DiagX(0:Nxp - 1)
  character(len=35) fname
  character(len=20) gname

  ! Mixing PDFs
  logical write_bins_flag
  real(rkind) pvd_thresh
  integer nbins_pdf, nbins_pdf_out
  real(rkind), allocatable :: pdf_bins(:)


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
    read (11, *) save_stats_dt_fine, base_time, nb_period
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
      if (save_movie_dt /= 0) then
        call init_chan_movie
      end if
    elseif (num_per_dir == 1) then
      stop 'Error: Duct not implemented!'
    elseif (num_per_dir == 0) then
      stop 'Error: Cavity not implemented!'
    end if

    ! Set time limit based on variables in input_chan
    Tb = int(100.d0 * 2.d0*4.*atan(1.0)/sqrt(N2) + 0.5)/100.d0 ! round to 2 decimal places
    if (check_flux) then
      time_limit = 1.d4
    else
      time_limit = (base_time + nb_period)*Tb
    end if
    fine_time = base_time*Tb
    save_stats_dt_fine = save_stats_dt_fine * Tb
    save_stats_dt = save_stats_dt * Tb

    if (rank == 0) then
      write(*,'("Buoyancy period T_b = ", ES12.5)') Tb
      write(*,'("Running until t = ", ES12.5)') fine_time
      write(*,'("Saving stats every ", ES12.5)') save_stats_dt
      write(*,'("Then, run until t = ", ES12.5)') time_limit
      write(*,'("Saving stats every ", ES12.5)') save_stats_dt_fine
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

    ! Find index and rank for vd_zmin
    ranky_vd_zmin = -1
    if (gyf(jstart) <= vd_zmin .and. gyf(jend + 1) > vd_zmin) then
      ranky_vd_zmin = rankY
      i = 1
      do while (.not. &
                (gyf(i) <= vd_zmin .and. gyf(i + 1) > vd_zmin))
        i = i + 1
      end do
      Ny_vd_zmin = i;
    end if
    if (rankY == ranky_vd_zmin) then
        write (*, *) "vd zmin index", Ny_vd_zmin
        write (*, *) "vd zmin rank", ranky_vd_zmin
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
    read (11, *) IC_type, kick, physical_noise
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
    read (11, *) r0, alpha_e, b0
    read (11, *)
    read (11, *) Lyc, Lyp, H, N2
    read (11, *)
    read (11, *) Svel_amp, Sb_amp, S_depth
    read (11, *)
    read (11, *) Svel_side_amp, S_side_depth
    read (11, *)
    read (11, *) Nb, Nphi, b_factor, phi_factor
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

    zvirt = -r0/(1.2d0 * alpha_e)
    F0 = (r0**2.d0) * b0

    ! Set up scatter plot arrays
    b_min = 0.d0
    b_max = b_factor * N2 * (LY - H)
    phi_min = 1.d-3
    phi_max = phi_factor * 5.d0*F0 / (3.d0*alpha_e) * ((0.9d0 * alpha_e * F0)**(-1.d0/3.d0)) * &
                              ((H + 5.d0*r0/(6.d0*alpha_e))**(-5.d0/3.d0))

    db = (b_max - b_min) / Nb
    dphi = (phi_max - phi_min) / Nphi

    vd_zmin = H - 0.25d0*(F0**(0.25d0)) * (N2**(-0.375d0))  
    !!! With current forcing strength, SL interface is pushed upwards!
    if (rank == 0) then
      write (*, *) "vd zmin", vd_zmin
      write (*, *) b_min, b_max, db, phi_min, phi_max, dphi
    end if

    
    Nb_out = int(ceiling(real(Nb)/NprocZ) * NprocZ)
    Nphi_out = int(ceiling(real(Nphi)/NprocZ) * NprocZ)
   
    allocate (bbins(1:Nb))
    allocate (phibins(1:Nphi))

    allocate (Ent_phi_flux(1:Nb_out, 1:Nphi_out))
    allocate (Ent_phi_flux_mem(1:Nb_out, 1:Nphi_out))
    allocate (Ent_phi_flux_cum(1:Nb_out, 1:Nphi_out))

    allocate (boundary_F(1:Nb_out, 1:Nphi_out))
    allocate (boundary_F_mem(1:Nb_out, 1:Nphi_out))
    allocate (boundary_F_cum(1:Nb_out, 1:Nphi_out))

    allocate (bbins_out(1:Nb_out))
    allocate (phibins_out(1:Nphi_out))
    allocate (weights(1:Nb, 1:Nphi))
    allocate (weights_vel(1:Nb, 1:Nphi))
    allocate (weights_flux(1:Nb, 1:Nphi))
    allocate (weights_flux_cum(1:Nb, 1:Nphi))
    allocate (weights_flux_mem(1:Nb, 1:Nphi))

    weights_flux_mem = 0.d0
    Ent_phi_flux_mem = 0.d0
    boundary_F_mem = 0.d0

    do i = 1, Nb
      bbins(i) = b_min + (i-0.5d0)*db
      bbins_out(i) = b_min + (i-0.5d0)*db
      if (rank == 0) write(*,*) "BBIN", i, bbins(i)
    end do

    do i = Nb + 1, Nb_out
      bbins_out(i) = -1.d0
    end do

    do i = 1, Nphi
      phibins(i) = phi_min + (i-0.5d0)*dphi
      phibins_out(i) = phi_min + (i-0.5d0)*dphi
      if (rank == 0) write(*,*) "PHIBIN", i, phibins(i)
    end do

    do i = Nphi + 1, Nphi_out
      phibins_out(i) = -1.d0
    end do

    ! For PDF arrays
    nbins_pdf_out = int(ceiling(real(nbins_pdf)/NprocZ) * NprocZ)

    ! Create PDF bins
    allocate(pdf_bins(0:nbins_pdf_out-1))

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
