
!----*|--.---------.---------.---------.---------.---------.---------.-|-------|
subroutine save_stats_chan(movie,final)
  !----*|--.---------.---------.---------.---------.---------.---------.-|-------|
  ! Computes domain-integrated and horizontally-integrated (X-Z) stats

  character(len=35) fname
  character(len=20) gname, gnamef
  logical movie,final
  integer i, j, k, n, l, m, phibin, bbin

  ! Buoyancy binning
  integer nbins, nbins_out
  real(rkind) bmin, bmax, db_pdf, zmin, zmax, dz_max
  real(rkind), allocatable :: bins(:)

  ! PDF binning
  real(rkind) pdf_min, pdf_max, dpdf

  ! Net diffusivity calculation
  real(rkind) dbdt_int, gradb2_int, kappa_net

  ! Scalar diagnostics
  real(rkind) thsum(0:Nyp + 1)
  real(rkind) thcount(0:Nyp + 1)
  ! Store/write 2D slices
  real(rkind) varxy(0:Nxm1, 1:Nyp), varzy(0:Nzp - 1, 1:Nyp), varxz(0:Nxm1, 0:Nzp - 1)

  ! HDF5 writing
  real(rkind) Diag(1:Nyp)
  real(rkind) DiagX(0:Nxp - 1)
  real(rkind) DiagPDF(0:int(nbins_pdf_out/NprocZ)-1)
  real(rkind) DiagB(1:int(Nb_out/NprocZ))
  real(rkind) DiagPhi(1:int(Nphi_out/NprocZ))

  if (rank == 0) &
    write (*, '("Saving Flow Statistics for Time Step       " I10)')  time_step

  call mpi_barrier(mpi_comm_world, ierror)
  call apply_BC_vel_mpi_post ! Apply BCs FIRST (it screws up ghost cells...)
  call apply_BC_th_mpi_post
  call ghost_chan_mpi
  call ghost_chan_mpi_j0 ! Need the j = 0 boundary filled for les output

  if (rank == 0) write (*, '("Time    = " ES12.5 "       dt = " ES12.5)') time, dt ! Note: dt is the physical / CFL-constrained time-step

  ! Store FF CUi in cr1(), and keep PP Ui in u1()
  do j = 0, Nyp + 1
    do k = 0, twoNkz
      do i = 0, Nxp - 1 ! Nkx
        cr1(i, k, j) = cu1(i, k, j)
        cr2(i, k, j) = cu2(i, k, j)
        cr3(i, k, j) = cu3(i, k, j)
        do n = 1, N_th
          crth(i, k, j, n) = cth(i, k, j, n)
        end do
      end do
    end do
  end do

  ! Convert to physical space
  call fft_xz_to_physical(cu1, u1)
  call fft_xz_to_physical(cu2, u2)
  call fft_xz_to_physical(cu3, u3)
  do n = 1, N_th
    call fft_xz_to_physical(cth(:, :, :, n), th(:, :, :, n))
  end do

  ! Compute ume(y), etc
  ! Also, computes Z-Averages ume_xy(x,y), etc, if not homogeneousX
  !   Otherwise, puts ume into ume_xy, etc
  call compute_averages(movie)


  !!! Dissipation Rate !!!
  call compute_TKE_diss(movie)
  call compute_MKE_diss(movie)
  if (use_LES) then
    call ghost_les_mpi ! Share nu_t
    call compute_TKE_diss_les
  end if
  
  ! Store epsilon
  tke_field = f1

  !!! TKE / RMS Velocities !!!
  call compute_TKE(movie)

  !!! Production and Reynolds Stresses !!!
  call compute_TKE_Production(movie)


  !!! MKE (on GY Grid) !!!
  do j = 2, Nyp
    mke(j) = 0.d0
    do k = 0, Nzp - 1
      do i = 0, Nxm1
        mke(j) = mke(j) &
                    +    (dyf(j - 1) * ume_xy(i, j)**2.d0 &
                         + dyf(j) * ume_xy(i, j - 1)**2.d0) / (2.d0 * dy(j)) &
                    +    (dyf(j - 1) * (wme_xy(i, j) + & ! Include the TWS!
                                    (1.d0 / (Ro_inv / delta)) * dTHdX(1) * (gyf(j) - 0.5d0*Ly))**2.d0 &
                         + dyf(j) * (wme_xy(i, j - 1) + &
                                         (1.d0 / (Ro_inv / delta)) * dTHdX(1) * (gyf(j - 1) - 0.5d0*Ly))**2.d0) / (2.d0 * dy(j)) &
                    +    vme_xy(i, j)**2.d0
      end do
    end do
  end do
  call mpi_allreduce(mpi_in_place, mke, Nyp + 2, mpi_double_precision, &
                     mpi_sum, mpi_comm_z, ierror)

  mke = 0.5 * mke / float(Nx * Nz)


  !!! Gradient of _Mean_ Velocity !!!
  do j = 1, Nyp
    dudy(j) = (ume(j) - ume(j - 1)) / (gyf(j) - gyf(j - 1))
    dwdy(j) = (wme(j) - wme(j - 1)) / (gyf(j) - gyf(j - 1))
  end do


  !!! Mean Square Shear !!!
  do j = 1, Nyp
    shear(j) = 0.d0
    do k = 0, Nzp - 1
      do i = 0, Nxm1
        f1(i, k, j) = ((u1(i, k, j + 1) - u1(i, k, j - 1)) / (2.d0 * dyf(j)))**2.d0 &
                    + ((u3(i, k, j + 1) - u3(i, k, j - 1)) / (2.d0 * dyf(j)))**2.d0
        shear(j) = shear(j) + f1(i, k, j)
      end do
    end do
  end do
  call mpi_allreduce(mpi_in_place, shear, Nyp + 2, mpi_double_precision, &
                     mpi_sum, mpi_comm_z, ierror)

  shear = shear / float(Nx * Nz)

  !gname = 'shear2_zstar'
  !call Bin_Ystar_and_Write(gname, f1)



  call compute_Vorticity(movie)



  !!! Write Mean Stats f(y) !!!
  fname = 'mean.h5'
  gname = 'time'
  call WriteHDF5_real(fname, gname, time)

  if (rankZ == 0) then

    gname = 'gzf'
    Diag = gyf(1:Nyp)
    call WriteStatH5_Y(fname, gname, Diag)

    gname = 'ume'
    Diag = ume(1:Nyp)
    call WriteStatH5_Y(fname, gname, Diag)

    gname = 'wme'
    Diag = vme(1:Nyp)
    call WriteStatH5_Y(fname, gname, Diag)

    gname = 'vme'
    Diag = wme(1:Nyp)
    call WriteStatH5_Y(fname, gname, Diag)

    gname = 'mke'
    Diag = mke(1:Nyp)
    call WriteStatH5_Y(fname, gname, Diag)

    gname = 'urms'
    Diag = urms(1:Nyp)
    call WriteStatH5_Y(fname, gname, Diag)

    gname = 'wrms'
    Diag = vrms(1:Nyp)
    call WriteStatH5_Y(fname, gname, Diag)

    gname = 'vrms'
    Diag = wrms(1:Nyp)
    call WriteStatH5_Y(fname, gname, Diag)

    gname = 'uw'
    Diag = uv(1:Nyp)
    call WriteStatH5_Y(fname, gname, Diag)

    gname = 'uv'
    Diag = uw(1:Nyp)
    call WriteStatH5_Y(fname, gname, Diag)

    gname = 'wv'
    Diag = wv(1:Nyp)
    call WriteStatH5_Y(fname, gname, Diag)

    gname = 'uu_dudx'
    Diag = uu_dudx(1:Nyp)
    call WriteStatH5_Y(fname, gname, Diag)

    gname = 'ww_dwdz'
    Diag = vv_dvdy(1:Nyp)
    call WriteStatH5_Y(fname, gname, Diag)

    gname = 'vu_dvdx'
    Diag = wu_dwdx(1:Nyp)
    call WriteStatH5_Y(fname, gname, Diag)

    gname = 'wu_dwdx'
    Diag = vu_dvdx(1:Nyp)
    call WriteStatH5_Y(fname, gname, Diag)

    gname = 'uw_dudz'
    Diag = uv_dudy(1:Nyp)
    call WriteStatH5_Y(fname, gname, Diag)

    gname = 'vw_dvdz'
    Diag = wv_dwdy(1:Nyp)
    call WriteStatH5_Y(fname, gname, Diag)

    gname = 'dudz'
    Diag = dudy(1:Nyp)
    call WriteStatH5_Y(fname, gname, Diag)

    gname = 'dvdz'
    Diag = dwdy(1:Nyp)
    call WriteStatH5_Y(fname, gname, Diag)

    gname = 'cp'
    Diag = dble(cp(0, 0, 1:Nyp))
    call WriteStatH5_Y(fname, gname, Diag)

    gname = 'shear'
    Diag = shear(1:Nyp)
    call WriteStatH5_Y(fname, gname, Diag)

    gname = 'omega_x'
    Diag = omega_x(1:Nyp)
    call WriteStatH5_Y(fname, gname, Diag)

    gname = 'omega_z'
    Diag = omega_y(1:Nyp)
    call WriteStatH5_Y(fname, gname, Diag)

    gname = 'omega_y'
    Diag = omega_z(1:Nyp)
    call WriteStatH5_Y(fname, gname, Diag)

  end if


  !!! Iterate through all TH Statistics !!!
  do n = 1, N_th
    ! Store FF CTH crth(), and keep PP in th() (Already done above)

    ! Compute the TH Gradients and store in CRi
    do j = 1, Nyp
      do k = 0, twoNkz
        do i = 0, Nxp - 1 ! Nkx
          ! Store gradients of TH(:,:,:,n) (if it is used) in CRi
          cr1(i, k, j) = cikx(i) * crth(i, k, j, n)
          cr2(i, k, j) = (crth(i, k, j + 1, n) - crth(i, k, j - 1, n)) / (gyf(j + 1) - gyf(j - 1))
          cr3(i, k, j) = cikz(k) * crth(i, k, j, n)
        end do
      end do
    end do

    ! Convert gradients to physical space
    call fft_xz_to_physical(cr1, r1)
    call fft_xz_to_physical(cr2, r2)
    call fft_xz_to_physical(cr3, r3)
    ! (Already have th in PP)

    !!! CWP(2022) net diffusivity calculation based on Penney et al. (2020) !!!
    ! th_mem stores buoyancy from previous time step for calculating time derivative
    dbdt_int = 0.d0
    gradb2_int = 0.d0

    do j = 1, Nyp
      do k = 0, Nzp - 1
        do i = 0, Nxm1
          if (gyf(j) > Lyc+Lyp) then
            dbdt_int = dbdt_int + ((th(i, k, j, n)**2.d0 - th_mem(i, k, j, n)**2.d0) / dt) * (dyf(j) * dx(1) * dz(1))
            gradb2_int = gradb2_int + (r1(i, k, j)**2.d0 &
                                       +  r2(i, k, j)**2.d0 &
                                       + (r3(i, k, j))**2.d0) * (dyf(j) * dx(1) * dz(1))
          end if
        end do
      end do
    end do

    call mpi_allreduce(mpi_in_place, dbdt_int, 1, mpi_double_precision, &
                     mpi_sum, mpi_comm_world, ierror)
    call mpi_allreduce(mpi_in_place, gradb2_int, 1, mpi_double_precision, &
                     mpi_sum, mpi_comm_world, ierror)

    kappa_net = -0.5d0*dbdt_int/gradb2_int

    fname = 'mean.h5'
    write (gname,'("kappa", I0.1 "_net")') n
    call WriteHDF5_real(fname, gname, kappa_net)


    !!! Write pointwise diapycnal velocity !!!
    ! e = dz/db * kappa * (grad^2 b)
    ! Store in e_field. Store second derivatives in s1, s2, s3.

    do j = 1, Nyp
      do k = 0, twoNkz
        do i = 0, Nxp - 1
          cs1(i, k, j) = -kx2(i) * crth(i, k, j, n)
          cs2(i, k, j) = (((crth(i, k, j + 1, n) - crth(i, k, j, n)) / dy(j+1)) - &
                          ((crth(i, k, j, n) - crth(i, k, j - 1, n)) / dy(j))) / &
                          dyf(j)
          cs3(i, k, j) = -kz2(k) * crth(i, k, j, n)
        end do
      end do
    end do
   
    ! Convert second derivatives to physical space
    call fft_xz_to_physical(cs1, s1)
    call fft_xz_to_physical(cs2, s2)
    call fft_xz_to_physical(cs3, s3)

    do j = 1, Nyp
      !thsum(j) = 0.d0
      !thcount(j) = 0.d0
      do k = 0, Nzp - 1
        do i = 0, Nxm1
          s4(i, k, j) =  s1(i, k, j) + s2(i, k, j) + s3(i, k, j) ! grad^2 b

          !! horizontal db/dz average within plume
          !if (th(i, k, j, 2) > 0.d0) then 
            !thsum(j) = thsum(j) + r2(i, k, j) 
            !thcount(j) = thcount(j) + 1.d0
          !end if
        end do
      end do
    end do

    !call mpi_allreduce(mpi_in_place, thsum, (Nyp + 2), &
                       !mpi_double_precision, mpi_sum, mpi_comm_z, ierror)
    !call mpi_allreduce(mpi_in_place, thcount, (Nyp + 2), &
                       !mpi_double_precision, mpi_sum, mpi_comm_z, ierror)

    !do j = 1, Nyp
      !thsum(j) = thsum(j) / thcount(j) !thsum now contains horizontal plume average of db/dz
    !end do
  
    do j = 1, Nyp
      do k = 0, Nzp - 1
        do i = 0, Nxm1
          e_field(i, k, j) = s4(i, k, j) * (nu*Pr(n) + kappa_t(i, k, j, n)) / r2(i, k, j)
        end do
      end do
    end do

    if (movie) then
      fname = 'movie.h5'
      call mpi_barrier(mpi_comm_world, ierror)
      if (rankZ == rankzmovie) then
        do i = 0, Nxm1
          do j = 1, Nyp
            varxy(i, j) = e_field(i, NzMovie, j)
          end do
        end do
        write (gname,'("diapycvel", I0.1 "_xz")') n
        call WriteHDF5_XYplane(fname, gname, varxy)
      end if
      call mpi_barrier(mpi_comm_world, ierror)
      if (rankY == rankymovie) then
        do i = 0, Nxm1
          do j = 0, Nzp - 1
            varxz(i, j) = e_field(i, j, NyMovie)
          end do
        end do
        write (gname,'("diapycvel", I0.1 "_xy")') n
        call WriteHDF5_XZplane(fname, gname, varxz)
      end if
      call mpi_barrier(mpi_comm_world, ierror)
      do i = 0, Nzp - 1
        do j = 1, Nyp
          varzy(i, j) = e_field(NxMovie, i, j)
        end do
      end do
      write (gname,'("diapycvel", I0.1 "_yz")') n
      call WriteHDF5_ZYplane(fname, gname, varzy)

    end if

    !!! RMS TH !!!
    thvar_xy = 0.
    do j = 1, Nyp
      thsum(j) = 0.
      do k = 0, Nzp - 1
        do i = 0, Nxm1
          thsum(j) = thsum(j) + (th(i, k, j, n) - thme_xy(i, j, n))**2.
          thvar_xy(i, j) = thvar_xy(i, j) + (th(i, k, j, n) - thme_xy(i, j, n))**2.
        end do
      end do
    end do
    call mpi_allreduce(mpi_in_place, thsum, (Nyp + 2), &
                       mpi_double_precision, mpi_sum, mpi_comm_z, ierror)

    thrms(:, n) = sqrt(thsum / float(Nx * Nz))
    thvar_xy = thvar_xy / float(Nz) ! Can't take sqrt, then sum next...

    if (n == 1 .and. movie .and. Nz > 1) then
      fname = 'mean_xz.h5'
      gname = 'thth_xz'
      call reduce_and_write_XYplane(fname, gname, thvar_xy, .false., movie)
    end if



    !!! TH Reynolds Stress, th*v -- i.e. Buoyancy Production !!!
    uvar_xy = 0.
    vvar_xy = 0.
    do j = 1, Nyp
      thsum(j) = 0.
      do k = 0, Nzp - 1
        do i = 0, Nxm1
          thsum(j) = thsum(j) + (th(i, k, j, n) - thme_xy(i, j, n)) &
                     * (0.5 * (u2(i, k, j) + u2(i, k, j + 1)) &
                        - 0.5 * (vme_xy(i, j) + vme_xy(i, j + 1)))

          uvar_xy(i, j) = uvar_xy(i, j) + (th(i, k, j, 1) - thme_xy(i, j, n)) &
                                        * (u1(i, k, j) - ume_xy(i, j))

          f1(i, k, j) = (th(i, k, j, 1) - thme_xy(i, j, n)) &
                                    * 0.5d0 * ((u2(i, k, j) - vme_xy(i, j)) &
                                             + (u2(i, k, j + 1) - vme_xy(i, j + 1)))
          vvar_xy(i, j) = vvar_xy(i, j) + f1(i, k, j)
        end do
      end do
    end do
    call mpi_allreduce(mpi_in_place, thsum, (Nyp + 2), &
                       mpi_double_precision, mpi_sum, mpi_comm_z, ierror)

    thv(:, n) = thsum / float(Nx * Nz)
    uvar_xy = uvar_xy / float(Nz)
    vvar_xy = vvar_xy / float(Nz)


    if (n == 1 .and. movie .and. Nz > 1) then
      fname = 'mean_xz.h5'
      gname = 'thu_xz'
      call reduce_and_write_XYplane(fname, gname, uvar_xy, .false., movie)
      gname = 'thw_xz'
      call reduce_and_write_XYplane(fname, gname, vvar_xy, .false., movie)
    end if

    !gname = 'thw_zstar'
    !call Bin_Ystar_and_Write(gname, f1)





    !!! TH Mean Production, th_m*v_m (with _full_ buoyancy) !!!
    do j = 1, Nyp
      thsum(j) = 0.
        do i = 0, Nxm1
          thsum(j) = thsum(j) + (thme_xy(i, j, n) + dTHdX(n)*(gx(i) - 0.5*Lx)) &
                     * (0.5 * (vme_xy(i, j) + vme_xy(i, j + 1)))
        end do
    end do

    thv_m(:, n) = thsum / float(Nx)



    !!! Gradient of _Mean_ TH !!!
    do j = 1, Nyp
      dthdy(j, n) = (thme(j, n) - thme(j - 1, n)) / (gyf(j) - gyf(j - 1))
    end do


    !!! PE Dissipation (Chi!), grad(TH) \cdot grad(TH) !!!
    ! Store |grad b|^2 in r1
    vvar_xy = 0.
    do j = 1, Nyp
      thsum(j) = 0.d0
      do k = 0, Nzp - 1
        do i = 0, Nxm1
          if ((gyf(j) > H).and.(n==1)) then
            r1(i, k, j) =  (r1(i, k, j) + dTHdX(n))**2.d0 &
                         !+  r2(i, k, j)**2.d0 &
                         +  (r2(i, k, j) - N2)**2.d0 &
                         + (r3(i, k, j) + dTHdZ(n))**2.d0
            vvar_xy(i, j) = vvar_xy(i, j) + r1(i, k, j)
            thsum(j)    = thsum(j) + r1(i, k, j)
          else
            r1(i, k, j) =  (r1(i, k, j) + dTHdX(n))**2.d0 &
                         +  r2(i, k, j)**2.d0 &
                         + (r3(i, k, j) + dTHdZ(n))**2.d0
            vvar_xy(i, j) = vvar_xy(i, j) + r1(i, k, j)
            thsum(j)    = thsum(j) + r1(i, k, j)
           end if
        end do
      end do
    end do
    call mpi_allreduce(mpi_in_place, thsum, (Nyp + 2), &
                       mpi_double_precision, mpi_sum, mpi_comm_z, ierror)

    pe_diss(:, n) = thsum / float(Nx * Nz) ! NOT actually PE dissipation -- just (grad TH)^2
    vvar_xy = vvar_xy / float(Nz)

    ! Write pointwise chi
    r1 = r1 * 2 * nu * Pr(1) / N2
    if (movie) then
  
      fname = 'movie.h5'
      call mpi_barrier(mpi_comm_world, ierror)
      if (rankZ == rankzmovie) then
        do i = 0, Nxm1
          do j = 1, Nyp
            varxy(i, j) = r1(i, NzMovie, j)
          end do
        end do
        write (gname,'("chi", I0.1 "_xz")') n
        call WriteHDF5_XYplane(fname, gname, varxy)
      end if
      call mpi_barrier(mpi_comm_world, ierror)
      if (rankY == rankymovie) then
        do i = 0, Nxm1
          do j = 0, Nzp - 1
            varxz(i, j) = r1(i, j, NyMovie)
          end do
        end do
        write (gname,'("chi", I0.1 "_xy")') n
        call WriteHDF5_XZplane(fname, gname, varxz)
      end if
      call mpi_barrier(mpi_comm_world, ierror)
      do i = 0, Nzp - 1
        do j = 1, Nyp
          varzy(i, j) = r1(NxMovie, i, j)
        end do
      end do
      write (gname,'("chi", I0.1 "_yz")') n
      call WriteHDF5_ZYplane(fname, gname, varzy)

    end if

    call mpi_barrier(mpi_comm_world, ierror)
    
    if (n == 1) then
      e_field = s4

      !!! CWP (2022) joint PDFs !!!
      ! s4 contains e
      ! r2 contains db/dz
      ! s5 contains epsilon
      ! s3 contains Ri

      ! compute log Re_b, store in Re_b_field
      ! compute LES corrected TKE, store in tke_field (which currently contains non-corrected TKE)
      do j = 1, Nyp
        do k = 0, Nzp - 1
          do i = 0, Nxm1
            chi_field(i, k, j) = log(r1(i, k, j))
            tke_field(i, k, j) = (nu + nu_t(i, k, j)) * tke_field(i, k, j) / nu
            Re_b_field(i, k, j) = log(tke_field(i, k, j) / ((nu + nu_t(i, k, j)) * abs(r2(i, k, j))))
            
            ! Now that LES-corrected TKE field has been computed, take log
            tke_field(i, k, j) = log(tke_field(i, k, j))
          end do
        end do
      end do

      ! compute Ri, store in Ri_field. Store u and v gradients in s1, s2.
      do j = 1, Nyp
        do k = 0, Nzp - 1
          do i = 0, Nxm1
            s1(i, k, j) = (u1(i, k, j) - u1(i, k, j - 1)) / dy(j)
            s2(i, k, j) = (u3(i, k, j) - u3(i, k, j - 1)) / dy(j)
            Ri_field(i, k, j) = 2.d0 * r2(i, k, j) / (s1(i, k, j)**2.d0 + s2(i, k, j)**2.d0)
          end do
        end do
      end do

      if (movie) then
        fname = 'movie.h5'
        call mpi_barrier(mpi_comm_world, ierror)
        if (rankZ == rankzmovie) then
          do j = 1, Nyp
            do i = 0, Nxm1
              varxy(i, j) = Ri_field(i, NzMovie, j)
            end do
          end do
          gname = 'Ri_xz'
          call WriteHDF5_XYplane(fname, gname, varxy)
        end if

        if (rankY == rankymovie) then
          do j = 0, Nzp - 1
            do i = 0, Nxm1
              varxz(i, j) = Ri_field(i, j, NyMovie)
            end do
          end do
          gname = 'Ri_xy'
          call WriteHDF5_XZplane(fname, gname, varxz)
        end if

        do j = 1, Nyp
          do i = 0, Nzp - 1
            varzy(i, j) = Ri_field(NxMovie, i, j)
          end do
        end do
        gname = 'Ri_yz'
        call WriteHDF5_ZYplane(fname, gname, varzy)
      end if

      if (movie) then
        fname = 'movie.h5'
        call mpi_barrier(mpi_comm_world, ierror)
        if (rankZ == rankzmovie) then
          do j = 1, Nyp
            do i = 0, Nxm1
              varxy(i, j) = tke_field(i, NzMovie, j)
            end do
          end do
          gname = 'tke_xz'
          call WriteHDF5_XYplane(fname, gname, varxy)
        end if

        if (rankY == rankymovie) then
          do j = 0, Nzp - 1
            do i = 0, Nxm1
              varxz(i, j) = tke_field(i, j, NyMovie)
            end do
          end do
          gname = 'tke_xy'
          call WriteHDF5_XZplane(fname, gname, varxz)
        end if

        do j = 1, Nyp
          do i = 0, Nzp - 1
            varzy(i, j) = tke_field(NxMovie, i, j)
          end do
        end do
        gname = 'tke_yz'
        call WriteHDF5_ZYplane(fname, gname, varzy)
      end if

      if (movie) then
        fname = 'movie.h5'
        call mpi_barrier(mpi_comm_world, ierror)
        if (rankZ == rankzmovie) then
          do j = 1, Nyp
            do i = 0, Nxm1
              varxy(i, j) = Re_b_field(i, NzMovie, j)
            end do
          end do
          gname = 'Re_b_xz'
          call WriteHDF5_XYplane(fname, gname, varxy)
        end if

        if (rankY == rankymovie) then
          do j = 0, Nzp - 1
            do i = 0, Nxm1
              varxz(i, j) = Re_b_field(i, j, NyMovie)
            end do
          end do
          gname = 'Re_b_xy'
          call WriteHDF5_XZplane(fname, gname, varxz)
        end if

        do j = 1, Nyp
          do i = 0, Nzp - 1
            varzy(i, j) = Re_b_field(NxMovie, i, j)
          end do
        end do
        gname = 'Re_b_yz'
        call WriteHDF5_ZYplane(fname, gname, varzy)
      end if


    end if
  

    !gname = 'chi_zstar'
    !call Bin_Ystar_and_Write(gname, r1)


    !if (n == 1) then
      !call compute_BPE

      !!! Compute integrated y*u1 at the left boundary !!!
      !do j = 1, Nyp
        !u1y_left(j) = 0.d0
        !do k = 0, Nzp - 1
          !u1y_left(j) = u1y_left(j) + gyf(j) * u1(0, k, j)
        !end do
      !end do
      !call mpi_allreduce(mpi_in_place, u1y_left, (Nyp + 2), &
                         !mpi_double_precision, mpi_sum, mpi_comm_z, ierror)

      !u1y_left = u1y_left / float(Nz)
      !if (rankZ == 0) then
        !call integrate_y_var(u1y_left, u1y_left_b)
      !end if

    !end if



    !!! Write Movie TH Slices !!!
    if (movie) then
      if (rank == 0) &
        write (*, '("Saving Movie Slice Output")')

      fname = 'movie.h5'
      call mpi_barrier(mpi_comm_world, ierror)
      if (rankZ == rankzmovie) then
        do j = 1, Nyp
          do i = 0, Nxm1
            varxy(i, j) = th(i, NzMovie, j, n)
          end do
        end do
        write (gname,'("th", I0.1 "_xz")') n
        call WriteHDF5_XYplane(fname, gname, varxy)
      end if

      if (rankY == rankymovie) then
        do j = 0, Nzp - 1
          do i = 0, Nxm1
            varxz(i, j) = th(i, j, NyMovie, n)
          end do
        end do
        write (gname,'("th", I0.1 "_xy")') n
        call WriteHDF5_XZplane(fname, gname, varxz)
      end if

      do j = 1, Nyp
        do i = 0, Nzp - 1
          varzy(i, j) = th(NxMovie, i, j, n)
        end do
      end do
      write (gname,'("th", I0.1 "_yz")') n
      call WriteHDF5_ZYplane(fname, gname, varzy)
    end if

  end do ! Over passive scalars, n
  
  !!! CWP 2022 azimuthal average calculations !!!
  ! For plume calculations, want variables on GXF, GYF, GZF grid to reduce loss of domain at centreline

  ! Interpolate vertical velocity onto vertical fractional grid
  call g2gf(u2)

  ! Convert to Fourier space
  call fft_xz_to_fourier(u1, cu1)
  call fft_xz_to_fourier(u2, cu2)
  call fft_xz_to_fourier(u3, cu3)
  call fft_xz_to_fourier(th(:, :, :, 1), cth(:, :, :, 1))
  call fft_xz_to_fourier(th(:, :, :, 2), cth(:, :, :, 2))
  ! p already in Fourier space

  ! Apply phase shift
  do j = 1, Nyp
    do k = 0, twoNkz
      do i = 0, Nxp - 1
        cs1(i,k,j) = exp(cikx(i) * dx(1)/2.d0 + cikz(k) * dz(1)/2.d0) * cu1(i,k,j)
        cs2(i,k,j) = exp(cikx(i) * dx(1)/2.d0 + cikz(k) * dz(1)/2.d0) * cu2(i,k,j)
        cs3(i,k,j) = exp(cikx(i) * dx(1)/2.d0 + cikz(k) * dz(1)/2.d0) * cu3(i,k,j)
        cs4(i,k,j) = exp(cikx(i) * dx(1)/2.d0 + cikz(k) * dz(1)/2.d0) * cth(i,k,j,1)
        cs5(i,k,j) = exp(cikx(i) * dx(1)/2.d0 + cikz(k) * dz(1)/2.d0) * cp(i,k,j)
        cs6(i,k,j) = exp(cikx(i) * dx(1)/2.d0 + cikz(k) * dz(1)/2.d0) * cth(i,k,j,2)
      end do
    end do
  end do

  ! Convert back to physical space
  call fft_xz_to_physical(cu1, u1)
  call fft_xz_to_physical(cu2, u2)
  call fft_xz_to_physical(cu3, u3)
  call fft_xz_to_physical(cth(:, :, :, 1), th(:, :, :, 1))
  call fft_xz_to_physical(cth(:, :, :, 2), th(:, :, :, 2))
  ! p already in Fourier space

  call fft_xz_to_physical(cs1, s1)
  call fft_xz_to_physical(cs2, s2)
  call fft_xz_to_physical(cs3, s3)
  call fft_xz_to_physical(cs4, s4)
  call fft_xz_to_physical(cs5, s5)
  call fft_xz_to_physical(cs6, s6)


  ! Move vertical velocity back to vertical full grid
  call gf2g(u2)

  !!! Compute radial and azimuthal velocity !!!
  do j = 1, Nyp
    do k = 0, Nzp - 1
      do i = 0, Nxm1
          ur(i, k, j) =  ( (gxf(i) - LX/2.d0) * s1(i, k, j) + (gzf(rankZ*Nzp+k) - LZ/2.d0) * s3(i, k, j) ) / &
                  sqrt( (gxf(i) - LX/2.d0)**2.d0 + (gzf(rankZ*Nzp+k) - LZ/2.d0)**2.d0 )
          utheta(i, k, j) =  ( (gxf(i) - LX/2.d0) * s3(i, k, j) - (gzf(rankZ*Nzp+k) - LZ/2.d0) * s1(i, k, j) ) / &
                  sqrt( (gxf(i) - LX/2.d0)**2.d0 + (gzf(rankZ*Nzp+k) - LZ/2.d0)**2.d0 )
      end do
    end do
  end do

  !!! Compute azimuthal averages !!!
  gname = 'th_az'
  call compute_azavg(gname, s6)

  gname = 'u_az'
  call compute_azavg_and_sfluc(gname, ur, u_sfluc)

  gname = 'v_az'
  call compute_azavg_and_sfluc(gname, utheta, v_sfluc)

  gname = 'w_az'
  call compute_azavg_and_sfluc(gname, s2, w_sfluc)

  gname = 'b_az'
  call compute_azavg_and_sfluc(gname, s4, b_sfluc)

  gname = 'p_az'
  call compute_azavg(gname, s5)

  !!! Compute spatial covariances !!!
  gname = 'uu_sfluc'
  s1 = u_sfluc * u_sfluc
  call compute_azavg(gname, s1)

  gname = 'uv_sfluc'
  s1 = u_sfluc * v_sfluc
  call compute_azavg(gname, s1)

  gname = 'uw_sfluc'
  s1 = u_sfluc * w_sfluc
  call compute_azavg(gname, s1)

  gname = 'ub_sfluc'
  s1 = u_sfluc * b_sfluc
  call compute_azavg(gname, s1)

  gname = 'vv_sfluc'
  s1 = v_sfluc * v_sfluc
  call compute_azavg(gname, s1)

  gname = 'vw_sfluc'
  s1 = v_sfluc * w_sfluc
  call compute_azavg(gname, s1)

  gname = 'ww_sfluc'
  s1 = w_sfluc * w_sfluc
  call compute_azavg(gname, s1)

  gname = 'wb_sfluc'
  s1 = w_sfluc * b_sfluc
  call compute_azavg(gname, s1)

  B_field = s1

  gname = 'bb_sfluc'
  s1 = b_sfluc * b_sfluc
  call compute_azavg(gname, s1)

  ! Save cross-sections of B_field
  if (movie) then
    fname = 'movie.h5'
    call mpi_barrier(mpi_comm_world, ierror)
    if (rankZ == rankzmovie) then
      do j = 1, Nyp
        do i = 0, Nxm1
          varxy(i, j) = B_field(i, NzMovie, j)
        end do
      end do
      write (gname,'("B_xz")')
      call WriteHDF5_XYplane(fname, gname, varxy)
    end if

    if (rankY == rankymovie) then
      do j = 0, Nzp - 1
        do i = 0, Nxm1
          varxz(i, j) = B_field(i, j, NyMovie)
        end do
      end do
      write (gname,'("B_xy")')
      call WriteHDF5_XZplane(fname, gname, varxz)
    end if

    do j = 1, Nyp
      do i = 0, Nzp - 1
        varzy(i, j) = B_field(NxMovie, i, j)
      end do
    end do
    write (gname,'("B_yz")')
    call WriteHDF5_ZYplane(fname, gname, varzy)
  end if

  s1 = th(:,:,:,2)
  s2 = th(:,:,:,1)

  !!! CWP(2022) tracer-density weighted scatter plot based on Penney et al. (2020) !!!

  gname = 'td_scatter'
  call tracer_density_weighting(gname, s2, s1, 0.95d0 * H, LY, weights)

  ! Write out scatter flux and corrected scatter
  if (rank == 0) then
    fname = 'movie.h5'
    gname = 'td_flux'
    call WriteHDF5_plane(fname, gname, weights_flux_cum) ! write cumulative flux (since last output) to file

    weights_flux_mem = weights_flux_mem + weights_flux_cum ! compute cumulative flux since t = 0
    weights_flux_cum = 0.d0

    weights = weights - weights_flux_mem ! compute SVD
    weights = weights / sum(weights_flux_mem)

    gname = 'svd'
    call WriteHDF5_plane(fname, gname, weights) ! write SVD

  end if

  ! Write out SVD bins
  fname = 'mean.h5'
  if ((write_bins_flag).and.(rankY == 0)) then

    gname = 'SVD_phibins'
    DiagPhi = phibins_out(1+rankZ * int(Nphi_out/NprocZ):(rankZ+1) * int(Nphi_out/NprocZ) )
    call WriteStatH5_X(fname, gname, DiagPhi, int(Nphi_out/NprocZ))

    gname = 'SVD_bbins'
    DiagB = bbins_out(1+rankZ * int(Nb_out/NprocZ):(rankZ+1) * int(Nb_out/NprocZ) )
    call WriteStatH5_X(fname, gname, DiagB, int(Nb_out/NprocZ))
  end if

  ! MPI barrier to ensure above weights calculation has been made. 'weights' constains SVD
  call mpi_barrier(mpi_comm_world, ierror)
  
  ! Communicate SVD to all cores
  call mpi_bcast(weights, Nb * Nphi, mpi_double_precision, 0, mpi_comm_world, ierror)


  !!! CWP (2023) mixing metric PDFs based on SVD class
  
  ! Compute volumes to be used as weights
  s3 = 0.d0
  do j = jstart_th(1), jend_th(1)
    do k = 0, Nzp - 1
      do i = 0, Nxm1
        s3(i, k, j) = dx(1) * dz(1) * dy(j)
      end do
    end do
  end do

  ! Identify SVD weight at each point
  svd_field = -1.d9
  s2 = -1.d9 ! just some number that will be excluded from PDF calculation...
  s4 = -1.d9
  s1 = -1.d9

  do j = jstart_th(1), jend_th(1)
    do k = 0, Nzp - 1
      do i = 0, Nxm1
        bbin = -1
        phibin = -1

        ! get b index
        if (th(i, k, j, 1) <= b_min) then 
          bbin = -1
        else if (th(i, k, j, 1) > b_max) then 
          bbin = -1
        else
          do l = 1, Nb ! b loop
            if ((th(i, k, j, 1) - bbins(l) > -0.5d0*db).and. &
                       (th(i, k, j, 1) - bbins(l) <= 0.5d0*db)) then
              bbin = l
            end if
          end do
        end if

        ! get phi index
        if (th(i, k, j, 2) <= phi_min) then 
          phibin = -1
        else if (th(i, k, j, 2) > phi_max) then
          phibin = -1
        else
          do m = 1, Nphi !phi loop
            if ((th(i, k, j, 2) - phibins(m) > -0.5d0*dphi).and.(th(i, k, j, 2) - phibins(m) <= 0.5d0*dphi)) then
              phibin = m
            end if
          end do
        end if
        
        if ((phibin > 0).and.(bbin > 0)) then
          svd_field(i, k, j) = weights(bbin, phibin) ! for output
          s2(i, k, j) = weights(bbin, phibin) ! for mixed
          s4(i, k, j) = -weights(bbin, phibin) ! for plume
          s1(i, k, j) = -abs(weights(bbin, phibin)) ! for mixing
        end if
      end do
    end do
  end do

  ! Create PDF bins

  do l = 1, 6

    select case (l)
    case (1)
      gnamef = 'Ri'
      pdf_min = Ri_min
      pdf_max = Ri_max
      pdf_field = Ri_field
    case (2)
      gnamef = 'Re_b'
      pdf_min = Re_b_min
      pdf_max = Re_b_max
      pdf_field = Re_b_field
    case (3)
      gnamef = 'chi'
      pdf_min = chi_min
      pdf_max = chi_max
      pdf_field = chi_field
    case (4)
      gnamef = 'tke'
      pdf_min = tke_min
      pdf_max = tke_max
      pdf_field = tke_field
    case (5)
      gnamef = 'e'
      pdf_min = e_min
      pdf_max = e_max
      pdf_field = e_field
    case (6)
      gnamef = 'B'
      pdf_min = BW_min
      pdf_max = BW_max
      pdf_field = B_field

    end select

    dpdf = (pdf_max - pdf_min) / (nbins_pdf - 1)

    do i = 0, nbins_pdf - 1
      pdf_bins(i) = pdf_min + i * dpdf
    end do

    do i = nbins_pdf, nbins_pdf_out - 1
      pdf_bins(i) = -1.d0
    end do
    
    if ((write_bins_flag).and.(rankY == 0)) then
      fname = 'mean.h5'
      gname = trim(gnamef)//'_pdf_bins'
      DiagPDF = pdf_bins(rankZ * int(nbins_pdf_out/NprocZ):(rankZ+1) * int(nbins_pdf_out/NprocZ) - 1)
      call WriteStatH5_X(fname, gname, DiagPDF, int(nbins_pdf_out/NprocZ))
    end if
   
    gname = trim(gnamef)//'_pdf_mixed'
    call Compute_PDF_SVD_and_Write(gname, s3, pdf_field, pdf_bins, s2, svd_thresh, 0.95d0*H, LY)

    gname = trim(gnamef)//'_pdf_plume'
    call Compute_PDF_SVD_and_Write(gname, s3, pdf_field, pdf_bins, s4, svd_thresh, 0.95d0*H, LY)

    gname = trim(gnamef)//'_pdf_mixing'
    call Compute_PDF_SVD_and_Write(gname, s3, pdf_field, pdf_bins, s1, -svd_thresh, 0.95d0*H, LY)

  end do 

  gname = 'chi_e_pdf'
  call Compute_JointPDF(gname, chi_field, chi_min, chi_max, nbins_pdf, e_field, e_min, e_max, nbins_pdf, &
          s3, s1, -svd_thresh, 0.95d0*H, LY)

  gname = 'chi_Ri_pdf'
  call Compute_JointPDF(gname, chi_field, chi_min, chi_max, nbins_pdf, Ri_field, Ri_min, Ri_max, nbins_pdf, &
          s3, s1, -svd_thresh, 0.95d0*H, LY)

  gname = 'e_Ri_pdf'
  call Compute_JointPDF(gname, e_field, e_min, e_max, nbins_pdf, Ri_field, Ri_min, Ri_max, nbins_pdf, &
          s3, s1, -svd_thresh, 0.95d0*H, LY)

  gname = 'chi_Reb_pdf'
  call Compute_JointPDF(gname, chi_field, chi_min, chi_max, nbins_pdf, Re_b_field, Re_b_min, Re_b_max, nbins_pdf, &
          s3, s1, -svd_thresh, 0.95d0*H, LY)

  gname = 'Ri_Reb_pdf'
  call Compute_JointPDF(gname, Ri_field, Ri_min, Ri_max, nbins_pdf, Re_b_field, Re_b_min, Re_b_max, nbins_pdf, &
          s3, s1, -svd_thresh, 0.95d0*H, LY)

  gname = 'Ri_B_pdf'
  call Compute_JointPDF(gname, Ri_field, Ri_min, Ri_max, nbins_pdf, B_field, BW_min, BW_max, nbins_pdf, &
          s3, s1, -svd_thresh, 0.95d0*H, LY)

  gname = 'chi_B_pdf'
  call Compute_JointPDF(gname, chi_field, chi_min, chi_max, nbins_pdf, B_field, BW_min, BW_max, nbins_pdf, &
          s3, s1, -svd_thresh, 0.95d0*H, LY)

  gname = 'e_B_pdf'
  call Compute_JointPDF(gname, e_field, e_min, e_max, nbins_pdf, B_field, BW_min, BW_max, nbins_pdf, &
          s3, s1, -svd_thresh, 0.95d0*H, LY)

  !!! Compute tracer vs. buoyancy distribution !!!
  
  ! Compute bins

  ! Find z values associated with above buoyancies, assuming linear density profile
  zmin = H
  zmax = H + b_max/N2
  dz_max = maxval(dz)
  nbins = floor((zmax - zmin)/dz_max)
  
  if ((rank == 0).and.(time == 0.d0)) write(*,*) "nbins", nbins
  nbins_out = int(ceiling(real(nbins)/NprocZ) * NprocZ)
  allocate(bins(0:nbins_out-1))

  db_pdf = (b_max - b_min)/(nbins-1)

  do i = 0, nbins - 1
    bins(i) = b_min + i*db_pdf
  end do
  
  do i = nbins, nbins_out - 1 ! Pad the useless part of the array with -1
    bins(i) = -1.d0
  end do

  s1 = th(:,:,:,2)
  s2 = th(:,:,:,1)

  !!! CWP (2022) buoyancy binning !!!

  gname = 'tb_source'
  call Compute_PDF_and_Write(gname, s1, s2, bins, H-0.05d0, H)

  gname = 'tb_strat'
  call Compute_PDF_and_Write(gname, s1, s2, bins, H, LY)

  !!! Write Mean TH Stats f(y) !!!
  fname = 'mean.h5'
  if (rankZ == 0) then
    do n = 1, N_th
      Diag = thme(1:Nyp, n)
      write (gname,'("thme", I0.2)') n
      call WriteStatH5_Y(fname, gname, Diag)

      Diag = dthdy(1:Nyp, n)
      write (gname,'("dthdz", I0.2)') n
      call WriteStatH5_Y(fname, gname, Diag)

      Diag = thrms(1:Nyp, n)
      write (gname,'("thrms", I0.2)') n
      call WriteStatH5_Y(fname, gname, Diag)

      Diag = thv(1:Nyp, n) ! \bar{w'b'}
      write (gname,'("thw", I0.2)') n
      call WriteStatH5_Y(fname, gname, Diag)

      Diag = thv_m(1:Nyp, n) ! \bar{w}\bar{b} + BG b!
      write (gname,'("thw", I0.2, "_m")') n
      call WriteStatH5_Y(fname, gname, Diag)

      Diag = pe_diss(1:Nyp, n)
      write (gname,'("pe_diss", I0.2)') n
      call WriteStatH5_Y(fname, gname, Diag)
    end do
  end if
  gname = 'u1z_x0'
  call WriteHDF5_real(fname, gname, u1y_left_b)




  !!! Write All Movie Slices !!!
  if (movie) then
    fname = 'movie.h5'
    call mpi_barrier(mpi_comm_world, ierror)
    if (rankZ == rankzmovie) then
      do j = 1, Nyp
        do i = 0, Nxm1
          varxy(i, j) = u1(i, NzMovie, j)
        end do
      end do
      gname = 'u_xz'
      call WriteHDF5_XYplane(fname, gname, varxy)
    end if

    call mpi_barrier(mpi_comm_world, ierror)
    if (rankZ == rankzmovie) then
      do j = 1, Nyp
        do i = 0, Nxm1
          ! Interpolate onto the GYF grid
          varxy(i, j) = 0.5 * (u2(i, NzMovie, j) + u2(i, NzMovie, j + 1))
        end do
      end do
      gname = 'w_xz'
      call WriteHDF5_XYplane(fname, gname, varxy)
    end if

    call mpi_barrier(mpi_comm_world, ierror)
    if (rankZ == rankzmovie) then
      do j = 1, Nyp
        do i = 0, Nxm1
          varxy(i, j) = u3(i, NzMovie, j)
        end do
      end do
      gname = 'v_xz'
      call WriteHDF5_XYplane(fname, gname, varxy)
    end if

    if (use_LES) then
      call mpi_barrier(mpi_comm_world, ierror)
      if (rankZ == rankzmovie) then
        do j = 1, Nyp
          do i = 0, Nxm1
            varxy(i, j) = nu_t(i, NzMovie, j)
          end do
        end do
        gname = 'nu_t_xz'
        call WriteHDF5_XYplane(fname, gname, varxy)
      end if
    end if

    call mpi_barrier(mpi_comm_world, ierror)
    if (rankY == rankymovie) then
      do j = 0, Nzp - 1
        do i = 0, Nxm1
          varxz(i, j) = u1(i, j, NyMovie)
        end do
      end do
      gname = 'u_xy'
      call WriteHDF5_XZplane(fname, gname, varxz)
    end if

    call mpi_barrier(mpi_comm_world, ierror)
    if (rankY == rankymovie) then
      do j = 0, Nzp - 1
        do i = 0, Nxm1
          varxz(i, j) = 0.5 * (u2(i, j, NyMovie) + u2(i, j, NyMovie + 1))
        end do
      end do
      gname = 'w_xy'
      call WriteHDF5_XZplane(fname, gname, varxz)
    end if

    call mpi_barrier(mpi_comm_world, ierror)
    if (rankY == rankymovie) then
      do j = 0, Nzp - 1
        do i = 0, Nxm1
          varxz(i, j) = u3(i, j, NyMovie)
        end do
      end do
      gname = 'v_xy'
      call WriteHDF5_XZplane(fname, gname, varxz)
    end if

    if (use_LES) then
      call mpi_barrier(mpi_comm_world, ierror)
      if (rankY == rankymovie) then
        do j = 0, Nzp - 1
          do i = 0, Nxm1
            varxz(i, j) = nu_t(i, j, NyMovie)
          end do
        end do
        gname = 'nu_t_xy'
        call WriteHDF5_XZplane(fname, gname, varxz)
      end if
    end if

    call mpi_barrier(mpi_comm_world, ierror)
    do j = 1, Nyp
      do i = 0, Nzp - 1
        varzy(i, j) = u1(NxMovie, i, j)
      end do
    end do
    gname = 'u_yz'
    call WriteHDF5_ZYplane(fname, gname, varzy)

    call mpi_barrier(mpi_comm_world, ierror)
    do j = 1, Nyp
      do i = 0, Nzp - 1
        varzy(i, j) = 0.5 * (u2(NxMovie, i, j) + u2(NxMovie, i, j + 1))
      end do
    end do
    gname = 'w_yz'
    call WriteHDF5_ZYplane(fname, gname, varzy)

    call mpi_barrier(mpi_comm_world, ierror)
    do j = 1, Nyp
      do i = 0, Nzp - 1
        varzy(i, j) = u3(NxMovie, i, j)
      end do
    end do
    gname = 'v_yz'
    call WriteHDF5_ZYplane(fname, gname, varzy)

    call mpi_barrier(mpi_comm_world, ierror)
    if (use_LES) then
      do j = 1, Nyp
        do i = 0, Nzp - 1
          varzy(i, j) = nu_t(NxMovie, i, j)
        end do
      end do
      gname = 'nu_t_yz'
      call WriteHDF5_ZYplane(fname, gname, varzy)
    end if

    call mpi_barrier(mpi_comm_world, ierror)
  end if ! END IF MOVIE







  ! Convert velocity back to Fourier space
  call fft_xz_to_fourier(u1, cu1)
  call fft_xz_to_fourier(u2, cu2)
  call fft_xz_to_fourier(u3, cu3)
  do n = 1, N_th
    call fft_xz_to_fourier(th(:, :, :, n), cth(:, :, :, n))
  end do




  !!! Spectrum of FT(u)^2 (Y-Averaged) !!!

  do j = 0, Nyp + 1
    do i = 0, Nxp - 1 ! Nkx
      cuu1_yx(j, i) = cu1(i, 0, j) ! Include both horizontal mean & structure of meanY
    end do
  end do

  cuu1_yx = cuu1_yx*conjg(cuu1_yx)

  DiagX = 0.
  do i = 0, Nxp - 1
    do j = 2, Nyp
      DiagX(i) = DiagX(i) + 0.5 * (cuu1_yx(j, i) + cuu1_yx(j - 1, i)) * dy(j) ! Integrate cuu1 in y
    end do
  end do
  call mpi_allreduce(mpi_in_place, DiagX, Nxp, &
                     mpi_double_precision, mpi_sum, mpi_comm_y, ierror)
  DiagX = DiagX / Ly

  if (rankY == 0) then
    fname = 'mean.h5'
    gname = 'FTx_uu'
    call WriteStatH5_X(fname, gname, DiagX, Nxp) ! Resulting size is Nx/2
  end if



  call mpi_barrier(mpi_comm_world, ierror)
  return
end

!----*|--.---------.---------.---------.---------.---------.---------.-|-------|
subroutine Compute_JointPDF(gname, X, Xmin, Xmax, NXbin, Y, Ymin, Ymax, NYbin, field, &
                con_field, con_thresh, zstart, zstop)
  !----*|--.---------.---------.---------.---------.---------.---------.-|-------|
  ! Calculates joint PDF of (X, Y) using field as weights, conditioning on con_field > con_thresh, and normalise
  ! CWP 2023

  character(len=20) gname
  real(rkind), pointer, intent(in) :: Y(:,:,:)
  real(rkind), pointer, intent(in) :: X(:,:,:)
  real(rkind), pointer, intent(in) :: field(:,:,:)
  real(rkind), pointer, intent(in) :: con_field(:,:,:)
  real(rkind) zstart, zstop, Xmin, Xmax, Ymin, Ymax, con_thresh
  integer NXbin, NYbin

  character(len=35) fname
  integer i, j, k, l, m, Xbin, Ybin
  real(rkind) Xbins(1:NXbin)
  real(rkind) Ybins(1:NYbin)
  real(rkind) dX, dY
  real(rkind) pdf(1:size(Xbins),1:size(Ybins))
  real(rkind) total_weight

  ! Construct bins
  dX = (Xmax - Xmin) / (NXbin - 1)

  do i = 1, NXbin
    Xbins(i) = Xmin + (i-1)*dX
  end do

  dY = (Ymax - Ymin) / (NYbin - 1)

  do i = 1, NYbin
    Ybins(i) = Ymin + (i-1)*dY
  end do


  pdf = 0.d0
  total_weight = 0.d0

  do j = jstart_th(1), jend_th(1)
    do k = 0, Nzp - 1
      do i = 0, Nxm1
        if (((gyf(j) <= zstop).and.(gyf(j) >= zstart)).and.(con_field(i, k, j) > con_thresh)) then
          Xbin = -1
          Ybin = -1

          do l = 1, NXbin-1
            if ((X(i, k, j) >= Xbins(l)).and.(X(i, k, j) < Xbins(l+1))) then
              Xbin = l
            end if
          end do

          do m = 1, NYbin-1
            if ((Y(i, k, j) >= Ybins(m)).and.(Y(i, k, j) < Ybins(m+1))) then
              Ybin = m
            end if
          end do

          if ((Xbin > 0).and.(Ybin > 0)) then
            pdf(Xbin, Ybin) = pdf(Xbin, Ybin) + field(i, k, j)
            total_weight = total_weight + field(i, k, j) * (Xbins(Xbin+1) - Xbins(Xbin)) * &
                    (Ybins(Ybin+1) - Ybins(Ybin))
          end if

        end if
      end do
    end do
  end do
    
  call mpi_allreduce(mpi_in_place, pdf, size(Xbins) * size(Ybins), mpi_double_precision, &
                     mpi_sum, mpi_comm_world, ierror)
  call mpi_allreduce(mpi_in_place, total_weight, 1, mpi_double_precision, &
                     mpi_sum, mpi_comm_world, ierror)

  if (total_weight > 0.d0) then !if total_weight = 0 then pdf = 0 also
    pdf = pdf / total_weight
  end if


  fname = 'movie.h5'
  if (rank == 0) then
    call WriteHDF5_plane(fname, gname, pdf)
  end if

  gname = trim(gname)//'_w'
  call WriteHDF5_real(fname, gname, total_weight)

end

!----*|--.---------.---------.---------.---------.---------.---------.-|-------|
subroutine tracer_density_weighting(gname, buoyancy, tracer, zstart, zstop, weights)
  !----*|--.---------.---------.---------.---------.---------.---------.-|-------|
  ! Calculates weights for (b, phi) scatter plot

  real(rkind), pointer, intent(in) :: buoyancy(:,:,:)
  real(rkind), pointer, intent(in) :: tracer(:,:,:)
  real(rkind), pointer, intent(inout) :: weights(:,:)
  real(rkind) zstart, zstop

  character(len=35) fname
  character(len=20) gname
  integer i, j, k, l, m
  integer bbin, phibin
  
  real(rkind) volume

  volume = 0.d0
  weights = 0.d0

  do j = jstart_th(1), jend_th(1)
    do k = 0, Nzp - 1
      do i = 0, Nxm1
        if ((gyf(j) <= zstop).and.(gyf(j) >= zstart)) then
          bbin = -1
          phibin = -1

          ! get b index
          if (buoyancy(i, k, j) <= b_min) then 
            bbin = -1
          else if (buoyancy(i, k, j) > b_max) then 
            bbin = -1
          else
            do l = 1, Nb ! b loop
              if ((buoyancy(i, k, j) - bbins(l) > -0.5d0*db).and. &
                         (buoyancy(i, k, j) - bbins(l) <= 0.5d0*db)) then
                bbin = l
              end if
            end do
          end if

          ! get phi index
          if (tracer(i, k, j) <= phi_min) then 
            phibin = -1
          else if (tracer(i, k, j) > phi_max) then
            phibin = -1
          else
            do m = 1, Nphi !phi loop
              if ((tracer(i, k, j) - phibins(m) > -0.5d0*dphi).and.(tracer(i, k, j) - phibins(m) <= 0.5d0*dphi)) then
                phibin = m
              end if
            end do
          end if

          if ((bbin > 0).and.(phibin > 0)) then
            weights(bbin, phibin) = weights(bbin, phibin) + (dy(j) * dx(1) * dz(1))
            volume = volume + (dy(j) * dx(1) * dz(1))
          end if
        end if
      end do
    end do
  end do
    
  call mpi_allreduce(mpi_in_place, weights, Nb * Nphi, mpi_double_precision, &
                     mpi_sum, mpi_comm_world, ierror)
  call mpi_allreduce(mpi_in_place, volume, 1, mpi_double_precision, &
                     mpi_sum, mpi_comm_world, ierror)

  weights = weights! / volume


  if (rank == 0) then
    fname = 'movie.h5'
    call WriteHDF5_plane(fname, gname, weights)
  end if

end

!----*|--.---------.---------.---------.---------.---------.---------.-|-------|
subroutine tracer_density_flux(buoyancy, tracer, vvel, Nlayer, weights)
  !----*|--.---------.---------.---------.---------.---------.---------.-|-------|
  ! Calculates weights for (b, phi) scatter plot

  real(rkind), pointer, intent(in) :: buoyancy(:,:,:)
  real(rkind), pointer, intent(in) :: tracer(:,:,:)
  real(rkind), pointer, intent(in) :: vvel(:,:,:)
  real(rkind), pointer, intent(inout) :: weights(:,:)
  real(rkind) zstart, zstop, volume

  character(len=35) fname
  character(len=20) gname
  integer i, j, k, l, m, Nlayer
  integer bbin, phibin
  
  volume = 0.d0
  weights = 0.d0

  if (rankY == rankymovie) then
    j = Nlayer
    do k = 0, Nzp - 1
      do i = 0, Nxm1
        bbin = -1
        phibin = -1
  
        ! get b index
        if (buoyancy(i, k, j) <= b_min) then 
          bbin = -1
        else if (buoyancy(i, k, j) > b_max) then 
          bbin = -1
        else
          do l = 1, Nb ! b loop
            if ((buoyancy(i, k, j) - bbins(l) > -0.5d0*db).and. &
                       (buoyancy(i, k, j) - bbins(l) <= 0.5d0*db)) then
              bbin = l
            end if
          end do
        end if
  
        ! get phi index
        if (tracer(i, k, j) <= phi_min) then 
          phibin = -1
        else if (tracer(i, k, j) > phi_max) then
          phibin = -1
        else
          do m = 1, Nphi !phi loop
            if ((tracer(i, k, j) - phibins(m) > -0.5d0*dphi).and.(tracer(i, k, j) - phibins(m) <= 0.5d0*dphi)) then
              phibin = m
            end if
          end do
        end if

        if ((bbin > 0).and.(phibin>0)) then
          weights(bbin, phibin) = weights(bbin, phibin) + (dy(j) * dx(1) * vvel(i, k, j) * dt)
        end if
        volume = volume + (dy(j) * dx(1) * dz(1))
      end do
    end do
  end if

  call mpi_allreduce(mpi_in_place, weights, Nb * Nphi, mpi_double_precision, &
                     mpi_sum, mpi_comm_world, ierror)
  call mpi_allreduce(mpi_in_place, volume, 1, mpi_double_precision, &
                     mpi_sum, mpi_comm_world, ierror)

  if (rank == 0) write(*,*) "flux volume", volume
end


!----*|--.---------.---------.---------.---------.---------.---------.-|-------|
subroutine compute_averages(movie)
  !----*|--.---------.---------.---------.---------.---------.---------.-|-------|
  ! Calculates horizontal averages
  !   and also z-line averages (if homogeneousX == .false.)
  ! NOTE: Assumes Ui are PP

  character(len=35) fname
  character(len=20) gname
  logical movie
  integer i, j, k, n
  real(rkind) ubulk


  !!! Horizontally-Averaged Velocities ume(y) (and Broadcast), and Write Bulk U !!!
  if (rankZ == 0) then
    ume = dble(cr1(0, 0, :))
    vme = dble(cr2(0, 0, :)) ! Still on GY
    wme = dble(cr3(0, 0, :))
    do n = 1, N_th
      thme(:, n) = dble(crth(0, 0, :, n))
    end do
  end if

  call integrate_y_var(ume, ubulk)
  if (rank == 0) write (*,  '("U Bulk  = " ES26.18)') ubulk

  call mpi_bcast(ume, Nyp + 2, mpi_double_precision, 0, &
                 mpi_comm_z, ierror)
  call mpi_bcast(vme, Nyp + 2, mpi_double_precision, 0, &
                 mpi_comm_z, ierror)
  call mpi_bcast(wme, Nyp + 2, mpi_double_precision, 0, &
                 mpi_comm_z, ierror)
  call mpi_bcast(thme, (Nyp + 2) * N_th, &
                 mpi_double_precision, 0, mpi_comm_z, ierror)


  if (Nz > 1 .and. .not. homogeneousX) then ! If 3D and if x-direction is not homogeneous...

    !!! XY Mean Velocities and Buoyancy (and keep) !!!
    do j = 1, Nyp
      do i = 0, Nxm1
        ume_xy(i, j) = 0.d0
        var_xy(i, j) = 0.d0
        wme_xy(i, j) = 0.d0
        do k = 0, Nzp - 1
          ume_xy(i, j) = ume_xy(i, j) + u1(i, k, j)
          var_xy(i, j) = var_xy(i, j) + 0.5d0 * (u2(i, k, j) + u2(i, k, j + 1)) ! Mean on GYF (to write out)
          wme_xy(i, j) = wme_xy(i, j) + u3(i, k, j)
        end do
      end do
    end do

    do n = 1, N_th
      do j = 1, Nyp
        do i = 0, Nxm1
          thme_xy(i, j, n) = 0.d0
          do k = 0, Nzp - 1
            thme_xy(i, j, n) = thme_xy(i, j, n) + th(i, k, j, n)
          end do
        end do
      end do
    end do

    vme_xy = 0.d0
    do j = 1, Nyp + 1
      do i = 0, Nxm1
        do k = 0, Nzp - 1
          vme_xy(i, j) = vme_xy(i, j) + u2(i, k, j) ! Mean on GY (To use in code)
        end do
      end do
    end do

    ume_xy = ume_xy / float(Nz)
    vme_xy = vme_xy / float(Nz)
    var_xy = var_xy / float(Nz)
    wme_xy = wme_xy / float(Nz)
    thme_xy = thme_xy / float(Nz)

    ! Use allreduce so that each process in comm_z has the mean
    call mpi_allreduce(mpi_in_place, vme_xy, Nx * (Nyp + 1), &
                       mpi_double_precision, mpi_sum, mpi_comm_z, ierror)

    fname = 'mean_xz.h5'

    gname = 'ume_xz'
    call reduce_and_write_XYplane(fname, gname, ume_xy, .true., movie)
    gname = 'wme_xz'
    call reduce_and_write_XYplane(fname, gname, var_xy, .false., movie)
    gname = 'vme_xz'
    call reduce_and_write_XYplane(fname, gname, wme_xy, .true., movie)
    gname = 'thme_xz'
    call reduce_and_write_XYplane(fname, gname, thme_xy(:,:,1), .true., movie)

    if (N_th > 1) then
      do n = 2, N_th
        call reduce_and_write_XYplane(fname, gname, thme_xy(:,:,n), .true., .false.)
      end do
    end if

  else
    ! Put ume into ume_xy, etc

    do j = 1, Nyp
      do i = 0, Nxm1
        ume_xy(i, j) = ume(j)
        wme_xy(i, j) = wme(j)
      end do
    end do
    do j = 1, Nyp + 1
      do i = 0, Nxm1
        vme_xy(i, j) = vme(j)
      end do
    end do
    do n = 1, N_th
      do j = 1, Nyp
        do i = 0, Nxm1
          thme_xy(i, j, n) = thme(j, n)
        end do
      end do
    end do

  end if

end



!----*|--.---------.---------.---------.---------.---------.---------.-|-------|
subroutine compute_TKE_diss(movie)
  !----*|--.---------.---------.---------.---------.---------.---------.-|-------|
  ! Calculate the turbulent dissipation rate, epsilon
  ! Note that this is actually the pseudo-dissipation (see Pope, Turb. Flows)
  ! for an explanation
  ! Assumes that FF Ui are stored in cri(), and PP Ui are in ui()

  character(len=35) fname
  character(len=20) gname
  logical movie
  real(rkind) Diag(1:Nyp)
  real(rkind) varxy(0:Nxm1, 1:Nyp), varzy(0:Nzp - 1, 1:Nyp), varxz(0:Nxm1, 0:Nzp - 1)
  integer i, j, k
  integer k_start

  ! Store the 3D dissipation rate in F1
  f1 = 0.d0

  ! Compute the turbulent dissipation rate, epsilon=nu*<du_i/ex_j du_i/dx_j>
  ! epsilon will be calculated on the GY grid (2:Nyp)
  !  This is so that it remains conserved (as in the code)
  epsilon = 0.

  if (.not. homogeneousX) then
    k_start = 1 ! Skip the mean component!
  else
    k_start = 0
  end if


  ! Compute du/dx  NOTE: Remove mean if not homogeneousX (k_start)
  if (k_start == 1) cs1 = 0
  do j = 1, Nyp
    do k = k_start, twoNkz
      do i = 0, Nxp - 1
        cs1(i, k, j) = cikx(i) * cr1(i, k, j)
      end do
    end do
  end do
  call fft_xz_to_physical(cs1, s1)
  do j = 2, Nyp
    do k = 0, Nzp - 1
      do i = 0, Nxm1
        epsilon(j) = epsilon(j) + (dyf(j - 1) * s1(i, k, j)**2.d0 &
                                   + dyf(j) * s1(i, k, j - 1)**2.d0) / (2.d0 * dy(j))
        f1(i, k, j) = f1(i, k, j) + (dyf(j - 1) * s1(i, k, j)**2.d0 &
                                     + dyf(j) * s1(i, k, j - 1)**2.d0) / (2.d0 * dy(j))
      end do
    end do
  end do


  ! Compute dv/dx  NOTE: Remove mean if not homogeneousX (k_start)
  if (k_start == 1) cs1 = 0
  do j = 2, Nyp
    do k = k_start, twoNkz
      do i = 0, Nxp - 1
        cs1(i, k, j) = cikx(i) * cr2(i, k, j)
      end do
    end do
  end do
  ! Convert to physical space
  call fft_xz_to_physical(cs1, s1)
  do j = 2, Nyp
    do k = 0, Nzp - 1
      do i = 0, Nxm1
        epsilon(j) = epsilon(j) + (s1(i, k, j)**2.0)
        f1(i, k, j) = f1(i, k, j) + (s1(i, k, j)**2.0)
      end do
    end do
  end do


  ! Compute du/dy at GY gridpoints  NOTE: Remove mean
  do j = 2, Nyp
    do k = 0, Nzp - 1
      do i = 0, Nxm1
        s1(i, k, j) = ((u1(i, k, j) - ume_xy(i, j)) &
                       - (u1(i, k, j - 1) - ume_xy(i, j - 1))) &
                      / dy(j)
      end do
    end do
  end do
  do j = 2, Nyp
    do k = 0, Nzp - 1
      do i = 0, Nxm1
        epsilon(j) = epsilon(j) + (s1(i, k, j)**2.0)
        f1(i, k, j) = f1(i, k, j) + (s1(i, k, j)**2.0)
      end do
    end do
  end do


  ! Compute dw/dx  NOTE: Remove mean if not homogeneousX (k_start)
  if (k_start == 1) cs1 = 0
  do j = 1, Nyp
    do k = k_start, twoNkz
      do i = 0, Nxp - 1
        cs1(i, k, j) = cikx(i) * cr3(i, k, j)
      end do
    end do
  end do
  call fft_xz_to_physical(cs1, s1)
  do j = 2, Nyp
    do k = 0, Nzp - 1
      do i = 0, Nxm1
        epsilon(j) = epsilon(j) + (dyf(j - 1) * s1(i, k, j)**2.d0 &
                                   + dyf(j) * s1(i, k, j - 1)**2.d0) / (2.d0 * dy(j))
        f1(i, k, j) = f1(i, k, j) + (dyf(j - 1) * s1(i, k, j)**2.d0 &
                                     + dyf(j) * s1(i, k, j - 1)**2.d0) / (2.d0 * dy(j))
      end do
    end do
  end do


  ! Compute du/dz at GY gridpoints
  do j = 1, Nyp
    do k = 0, twoNkz
      do i = 0, Nxp - 1
        cs1(i, k, j) = cikz(k) * cr1(i, k, j)
      end do
    end do
  end do
  call fft_xz_to_physical(cs1, s1)
  do j = 2, Nyp
    do k = 0, Nzp - 1
      do i = 0, Nxm1
        epsilon(j) = epsilon(j) + (dyf(j - 1) * s1(i, k, j)**2.d0 &
                                   + dyf(j) * s1(i, k, j - 1)**2.d0) / (2.d0 * dy(j))
        f1(i, k, j) = f1(i, k, j) + (dyf(j - 1) * s1(i, k, j)**2.d0 &
                                     + dyf(j) * s1(i, k, j - 1)**2.d0) / (2.d0 * dy(j))
      end do
    end do
  end do


  ! Compute dv/dy at GY gridpoints  NOTE: Remove mean
  do j = 2, Nyp
    do k = 0, Nzp - 1
      do i = 0, Nxm1
        s1(i, k, j) = ((u2(i, k, j + 1) - vme_xy(i, j + 1)) &
                       - (u2(i, k, j - 1) - vme_xy(i, j - 1))) &
                      / (gy(j + 1) - gy(j - 1))
      end do
    end do
  end do
  do j = 2, Nyp
    do k = 0, Nzp - 1
      do i = 0, Nxm1
        epsilon(j) = epsilon(j) + (s1(i, k, j)**2.0)
        f1(i, k, j) = f1(i, k, j) + (s1(i, k, j)**2.0)
      end do
    end do
  end do


  ! Compute dw/dy at GY gridpoints  NOTE: Remove mean
  do j = 2, Nyp
    do k = 0, Nzp - 1
      do i = 0, Nxm1
        s1(i, k, j) = ((u3(i, k, j) - wme_xy(i, j)) &
                       - (u3(i, k, j - 1) - wme_xy(i, j - 1))) &
                      / dy(j)
      end do
    end do
  end do
  do j = 2, Nyp
    do k = 0, Nzp - 1
      do i = 0, Nxm1
        epsilon(j) = epsilon(j) + (s1(i, k, j)**2.0)
        f1(i, k, j) = f1(i, k, j) + (s1(i, k, j)**2.0)
      end do
    end do
  end do


  ! Compute dv/dz
  do j = 2, Nyp
    do k = 0, twoNkz
      do i = 0, Nxp - 1
        cs1(i, k, j) = cikz(k) * cr2(i, k, j)
      end do
    end do
  end do
  call fft_xz_to_physical(cs1, s1)
  do j = 2, Nyp
    do k = 0, Nzp - 1
      do i = 0, Nxm1
        epsilon(j) = epsilon(j) + (s1(i, k, j)**2.0)
        f1(i, k, j) = f1(i, k, j) + (s1(i, k, j)**2.0)
      end do
    end do
  end do


  ! Compute dw/dz
  do j = 1, Nyp
    do k = 0, twoNkz
      do i = 0, Nxp - 1
        cs1(i, k, j) = cikz(k) * cr3(i, k, j)
      end do
    end do
  end do
  call fft_xz_to_physical(cs1, s1)
  do j = 2, Nyp
    do k = 0, Nzp - 1
      do i = 0, Nxm1
        epsilon(j) = epsilon(j) + (dyf(j - 1) * s1(i, k, j)**2.d0 &
                                   + dyf(j) * s1(i, k, j - 1)**2.d0) / (2.d0 * dy(j))
        f1(i, k, j) = f1(i, k, j) + (dyf(j - 1) * s1(i, k, j)**2.d0 &
                                     + dyf(j) * s1(i, k, j - 1)**2.d0) / (2.d0 * dy(j))
      end do
    end do
  end do


  epsilon = nu * epsilon / float(Nx * Nz)
  f1 = nu * f1
  call mpi_allreduce(mpi_in_place, epsilon, Nyp + 2, mpi_double_precision, &
                     mpi_sum, mpi_comm_z, ierror)




  ! Write mean / movie / mean_xy
  fname = 'mean.h5'
  if (rankZ == 0) then
    gname = 'epsilon'
    Diag = epsilon(1:Nyp)
    call WriteStatH5_Y(fname, gname, Diag)
  end if

  !gname = 'epsilon_zstar'
  !call Bin_Ystar_and_Write(gname, f1)


  if (movie) then

    fname = 'movie.h5'
    call mpi_barrier(mpi_comm_world, ierror)
    if (rankZ == rankzmovie) then
      do i = 0, Nxm1
        do j = 1, Nyp
          varxy(i, j) = f1(i, NzMovie, j)
        end do
      end do
      gname = 'epsilon_xz'
      call WriteHDF5_XYplane(fname, gname, varxy)
    end if
    call mpi_barrier(mpi_comm_world, ierror)
    if (rankY == rankymovie) then
      do i = 0, Nxm1
        do j = 0, Nzp - 1
          varxz(i, j) = f1(i, j, NyMovie)
        end do
      end do
      gname = 'epsilon_xy'
      call WriteHDF5_XZplane(fname, gname, varxz)
    end if
    call mpi_barrier(mpi_comm_world, ierror)
    do i = 0, Nzp - 1
      do j = 1, Nyp
        varzy(i, j) = f1(NxMovie, i, j)
      end do
    end do
    gname = 'epsilon_yz'
    call WriteHDF5_ZYplane(fname, gname, varzy)


    ! Mean XY Slice
    do j = 1, Nyp
      do i = 0, Nxm1
        uvar_xy(i, j) = 0.d0
        do k = 0, Nzp - 1
          uvar_xy(i, j) = uvar_xy(i, j) + f1(i, k, j)
        end do
      end do
    end do
    uvar_xy = uvar_xy / float(Nz)

    fname = 'mean_xz.h5'
    gname = 'epsilon_xz'
    if (movie) &
      call reduce_and_write_XYplane(fname, gname, uvar_xy, .false., movie)

  end if

  return
end



!----*|--.---------.---------.---------.---------.---------.---------.-|-------|
subroutine compute_MKE_diss(movie)
  !----*|--.---------.---------.---------.---------.---------.---------.-|-------|
  ! Calculate the mean dissipation rate, epsilon_m
  ! Note that this is actually the pseudo-dissipation (see Pope, Turb. Flows)
  ! Assumes that FF Ui are stored in cri(), and PP Ui are in ui()

  character(len=35) fname
  character(len=20) gname
  logical movie
  real(rkind) Diag(1:Nyp)
  real(rkind) varxy(0:Nxm1, 1:Nyp), tempxy(0:Nxm1, 1:Nyp)
  integer i, j

  ! Store the 2D dissipation rate in varxy
  varxy = 0.

  ! Compute the mean dissipation rate, epsilon_m=nu*<du_i/dx_j du_i/dx_j>
  ! epsilon will be calculated on the GY grid (2:Nyp)
  epsilon_m = 0.

  if (homogeneousX) then
    return
  end if


  ! Compute du/dx
  cs1 = 0
  do j = 1, Nyp
    do i = 0, Nxp - 1
      cs1(i, 0, j) = cikx(i) * cr1(i, 0, j)
    end do
  end do
  call fft_xz_to_physical(cs1, s1)
  do j = 1, Nyp
    do i = 0, Nxm1
      epsilon_m(j) = epsilon_m(j) + (dyf(j - 1) * s1(i, 0, j)**2.d0 &
                                 + dyf(j) * s1(i, 0, j - 1)**2.d0) / (2.d0 * dy(j))
      varxy(i, j) = varxy(i, j) + (dyf(j - 1) * s1(i, 0, j)**2.d0 &
                                   + dyf(j) * s1(i, 0, j - 1)**2.d0) / (2.d0 * dy(j))

      dudx_m(i, j) = s1(i, 0, j) ! Store for compute_TKE_Production
    end do
  end do


  ! Compute dv/dx
  cs1 = 0
  do j = 1, Nyp
    do i = 0, Nxp - 1
      cs1(i, 0, j) = cikx(i) * cr2(i, 0, j)
    end do
  end do
  ! Convert to physical space
  call fft_xz_to_physical(cs1, s1)
  do j = 1, Nyp
    do i = 0, Nxm1
      epsilon_m(j) = epsilon_m(j) + (s1(i, 0, j)**2.0)
      varxy(i, j) = varxy(i, j) + (s1(i, 0, j)**2.0)

      dvdx_m(i, j) = s1(i, 0, j) ! Store for compute_TKE_Production
    end do
  end do


  ! Compute du/dy at GY gridpoints
  do j = 2, Nyp
    do i = 0, Nxm1
      tempxy(i, j) = (ume_xy(i, j) - ume_xy(i, j - 1))  / dy(j)
    end do
  end do
  do j = 2, Nyp
    do i = 0, Nxm1
      epsilon_m(j) = epsilon_m(j) + (tempxy(i, j)**2.0)
      varxy(i, j) = varxy(i, j) + (tempxy(i, j)**2.0)
    end do
  end do


  ! Compute dw/dx
  cs1 = 0
  do j = 1, Nyp
    do i = 0, Nxp - 1
      cs1(i, 0, j) = cikx(i) * cr3(i, 0, j)
    end do
  end do
  call fft_xz_to_physical(cs1, s1)
  do j = 1, Nyp
    do i = 0, Nxm1
      epsilon_m(j) = epsilon_m(j) + (dyf(j - 1) * s1(i, 0, j)**2.d0 &
                                 + dyf(j) * s1(i, 0, j - 1)**2.d0) / (2.d0 * dy(j))
      varxy(i, j) = varxy(i, j) + (dyf(j - 1) * s1(i, 0, j)**2.d0 &
                                   + dyf(j) * s1(i, 0, j - 1)**2.d0) / (2.d0 * dy(j))

      dwdx_m(i, j) = s1(i, 0, j) ! Store for compute_TKE_Production
    end do
  end do


  ! Compute dv/dy at GY gridpoints
  do j = 2, Nyp
    do i = 0, Nxm1
      tempxy(i, j) = (vme_xy(i, j + 1) - vme_xy(i, j - 1)) &
                    / (gy(j + 1) - gy(j - 1))
    end do
  end do
  do j = 2, Nyp
    do i = 0, Nxm1
      epsilon_m(j) = epsilon_m(j) + (tempxy(i, j)**2.0)
      varxy(i, j) = varxy(i, j) + (tempxy(i, j)**2.0)
    end do
  end do


  ! Compute dw/dy at GY gridpoints
  do j = 2, Nyp
    do i = 0, Nxm1
      tempxy(i, j) = (wme_xy(i, j) - wme_xy(i, j - 1)) / dy(j) + &
                                (1.d0 / (Ro_inv / delta)) * dTHdX(1) ! Include TWS!
    end do
  end do
  do j = 2, Nyp
    do i = 0, Nxm1
      epsilon_m(j) = epsilon_m(j) + (tempxy(i, j)**2.0)
      varxy(i, j) = varxy(i, j) + (tempxy(i, j)**2.0)
    end do
  end do





  epsilon_m = nu * epsilon_m / float(Nx)
  varxy = nu * varxy


  ! Write mean / movie / mean_xy
  fname = 'mean.h5'
  if (rankZ == 0) then
    gname = 'epsilon_m'
    Diag = epsilon_m(1:Nyp)
    call WriteStatH5_Y(fname, gname, Diag)
  end if

  if (movie) then
    fname = 'mean_xz.h5'
    gname = 'epsilon_m_xz'
    if (rankZ == 0) then
      call WriteHDF5_XYplane(fname, gname, varxy)
    end if
  end if

  return
end



!----*|--.---------.---------.---------.---------.---------.---------.-|-------|
subroutine compute_TKE(movie)
  !----*|--.---------.---------.---------.---------.---------.---------.-|-------|
  ! Compute the RMS velocities
  !  Both the horizontal averages -- ume(y), etc
  !   and the z-averages -- ume_xz(x,y), etc
  ! Assumes PP Ui

  character(len=35) fname
  character(len=20) gname
  logical movie
  integer i, j, k



  !!! TKE f(y) & f(x,y) !!!
  ! urms and wrms are on the GYF grid
  ! vrms is defined on the GY grid
  uu_xy = 0.
  vv_xy = 0.
  ww_xy = 0.
  f1 = 0.

  do j = 1, Nyp
    urms(j) = 0.
    vrms(j) = 0.
    wrms(j) = 0.
    do k = 0, Nzp - 1
      do i = 0, Nxm1

        f1(i, k, j) = 0.5d0 * (u1(i, k, j) - ume_xy(i, j))**2. &
                    + 0.5d0 * (u2(i, k, j) - vme_xy(i, j))**2. &
                    + 0.5d0 * (u3(i, k, j) - wme_xy(i, j))**2.

        urms(j) = urms(j) + (u1(i, k, j) - ume_xy(i, j))**2.
        uu_xy(i, j) = uu_xy(i, j) + (u1(i, k, j) - ume_xy(i, j))**2.

        vrms(j) = vrms(j) + (u2(i, k, j) - vme_xy(i, j))**2.
        vv_xy(i, j) = vv_xy(i, j) + (u2(i, k, j) - vme_xy(i, j))**2.

        wrms(j) = wrms(j) + (u3(i, k, j) - wme_xy(i, j))**2.
        ww_xy(i, j) = ww_xy(i, j) + (u3(i, k, j) - wme_xy(i, j))**2.

      end do
    end do
  end do


  ! Communicate Horizontal Mean (Save them later)

  call mpi_allreduce(mpi_in_place, urms, Nyp+1, mpi_double_precision, &
                     mpi_sum, mpi_comm_z, ierror)
  call mpi_allreduce(mpi_in_place, vrms, Nyp+1, mpi_double_precision, &
                     mpi_sum, mpi_comm_z, ierror)
  call mpi_allreduce(mpi_in_place, wrms, Nyp+1, mpi_double_precision, &
                     mpi_sum, mpi_comm_z, ierror)

  urms = urms / float(Nx * Nz)
  vrms = vrms / float(Nx * Nz)
  wrms = wrms / float(Nx * Nz) ! Only take sqrt after summing across procs

  ! Get the bulk RMS value
  call integrate_y_var(urms, urms_b)
  call integrate_y_var(vrms, vrms_b)
  call integrate_y_var(wrms, wrms_b)
  ! Write out the bulk RMS Velocity
  if (rank == 0) then
    write (*,  '("<U_rms> = " ES26.18)') sqrt(urms_b)
    write (*,  '("<V_rms> = " ES26.18)') sqrt(wrms_b)
    write (*,  '("<W_rms> = " ES26.18)') sqrt(vrms_b)
  end if

  urms = sqrt(urms)
  vrms = sqrt(vrms)
  wrms = sqrt(wrms)



  ! Communicate Z-Mean

  uu_xy = uu_xy / float(Nz) ! Can't take sqrt, then sum next...
  vv_xy = vv_xy / float(Nz)
  ww_xy = ww_xy / float(Nz)

  if (Nz > 1) then

    ! Need these in rankZ = 0 to compute Production!
    fname = 'mean_xz.h5'
    gname = 'uu_xz'
    call reduce_and_write_XYplane(fname, gname, uu_xy, .false., movie)
    gname = 'ww_xz'
    call reduce_and_write_XYplane(fname, gname, vv_xy, .false., movie)
    gname = 'vv_xz'
    call reduce_and_write_XYplane(fname, gname, ww_xy, .false., movie)

  end if


  !gname = 'TKE_zstar'
  !call Bin_Ystar_and_Write(gname, f1)

  !f1 = 0.5d0 * (u1**2. + u2**2. + u3**2.)
  !gname = 'KE_zstar'
  !call Bin_Ystar_and_Write(gname, f1)



end




!----*|--.---------.---------.---------.---------.---------.---------.-|-------|
subroutine compute_TKE_Production(movie)
  !----*|--.---------.---------.---------.---------.---------.---------.-|-------|
  ! Compute the TKE Production terms, and also Reynolds Stresses
  !  Both the horizontal averages -- ume(y), etc
  !   and the z-averages -- ume_xz(x,y), etc
  ! Assumes PP Ui
  ! Requires dudx_m to be stored already from compute_MKE_diss

  character(len=35) fname
  character(len=20) gname
  logical movie
  integer i, j, k


  !!! Reynolds Stress !!!
  ! uv and wv are defined on the GY grid -- Then can compute that production conservatively!
  ! uw is defined on the GYF grid
  uvar_xy = 0. ! uv
  wvar_xy = 0. ! wv
  vvar_xy = 0. ! uw

  do j = 1, Nyp ! On GYF
    uw(j) = 0.
    do k = 0, Nzp - 1
      do i = 0, Nxm1

        uw(j) = uw(j) + (u1(i, k, j) - ume_xy(i, j)) &
                      * (u3(i, k, j) - wme_xy(i, j))
        vvar_xy(i, j) = vvar_xy(i, j) + (u1(i, k, j) - ume_xy(i, j)) &
                                      * (u3(i, k, j) - wme_xy(i, j))

      end do
    end do
  end do

  do j = 2, Nyp ! On GY
    uv(j) = 0.
    wv(j) = 0.
    do k = 0, Nzp - 1
      do i = 0, Nxm1

        uv(j) = uv(j) +   (dyf(j - 1) * (u1(i, k, j) - ume_xy(i, j)) + &
                            dyf(j) * (u1(i, k, j - 1) - ume_xy(i, j - 1))) &
                                    / (2.d0 * dy(j)) &
                              * (u2(i, k, j) - vme_xy(i, j))
        uvar_xy(i, j) = uvar_xy(i, j) +   (dyf(j - 1) * (u1(i, k, j) - ume_xy(i, j)) + &
                            dyf(j) * (u1(i, k, j - 1) - ume_xy(i, j - 1))) &
                                    / (2.d0 * dy(j)) &
                              * (u2(i, k, j) - vme_xy(i, j))

        wv(j) = wv(j) +   (dyf(j - 1) * (u3(i, k, j) - wme_xy(i, j)) + &
                            dyf(j) * (u3(i, k, j - 1) - wme_xy(i, j - 1))) &
                                    / (2.d0 * dy(j)) &
                              * (u2(i, k, j) - vme_xy(i, j))
        wvar_xy(i, j) = wvar_xy(i, j) +   (dyf(j - 1) * (u3(i, k, j) - wme_xy(i, j)) + &
                            dyf(j) * (u3(i, k, j - 1) - wme_xy(i, j - 1))) &
                                    / (2.d0 * dy(j)) &
                              * (u2(i, k, j) - vme_xy(i, j))

      end do
    end do
  end do



  ! Communicate Horizontal Mean (Save them later)

  call mpi_allreduce(mpi_in_place, uv, Nyp + 2, mpi_double_precision, &
                     mpi_sum, mpi_comm_z, ierror)
  call mpi_allreduce(mpi_in_place, wv, Nyp + 2, mpi_double_precision, &
                     mpi_sum, mpi_comm_z, ierror)
  call mpi_allreduce(mpi_in_place, uw, Nyp + 2, mpi_double_precision, &
                     mpi_sum, mpi_comm_z, ierror)


  uv = uv / float(Nx * Nz)
  wv = wv / float(Nx * Nz)
  uw = uw / float(Nx * Nz)



  ! Communicate Z-Mean

  uvar_xy = uvar_xy / float(Nz)
  wvar_xy = wvar_xy / float(Nz)
  vvar_xy = vvar_xy / float(Nz)

  if (Nz > 1) then
    fname = 'mean_xz.h5'
    gname = 'uw_xz'
    call reduce_and_write_XYplane(fname, gname, uvar_xy, .false., movie)
    gname = 'wv_xz'
    call reduce_and_write_XYplane(fname, gname, wvar_xy, .false., movie)
    gname = 'uv_xz'
    call reduce_and_write_XYplane(fname, gname, vvar_xy, .false., movie)


    !!! Production Terms (Just on rankZ = 0) !!!
    ! Just multiply the Reynolds Stress with the mean_xy field...
    !   Only compute as f(y), since can get z-average by multiplication...
    ! Use the mean field gradients stored from compute_MKE_diss
    if (rankZ == 0) then

      do j = 1, Nyp
        uu_dudx(j) = 0.
        wu_dwdx(j) = 0.
        do i = 0, Nxm1

          ! On GYF
          uu_dudx(j) = uu_dudx(j) +  uu_xy(i, j) * dudx_m(i, j)
          wu_dwdx(j) = wu_dwdx(j) +  vvar_xy(i, j)     * dwdx_m(i, j)

        end do
      end do

      do j = 2, Nyp
        vu_dvdx(j) = 0.
        vv_dvdy(j) = 0.
        uv_dudy(j) = 0.
        wv_dwdy(j) = 0.
        do i = 0, Nxm1

          ! On GY
          vu_dvdx(j) = vu_dvdx(j) +  uvar_xy(i, j)     * dvdx_m(i, j)

          vv_dvdy(j) = vv_dvdy(j) +  (vme_xy(i, j + 1) - vme_xy(i, j - 1)) &
                                           / (gy(j + 1) - gy(j - 1)) &
                                    * vv_xy(i, j)
          uv_dudy(j) = uv_dudy(j) +  (ume_xy(i, j) - ume_xy(i, j - 1)) &
                                           / dy(j) &
                                    * uvar_xy(i, j)
          wv_dwdy(j) = wv_dwdy(j) +  ((wme_xy(i, j) - wme_xy(i, j - 1)) &
                                           / dy(j)  + &
                                              (1.d0 / (Ro_inv / delta)) * dTHdX(1) )  &  ! Include the TWS!
                                    * wvar_xy(i, j)

        end do
      end do

      uu_dudx = uu_dudx / float(Nx)
      wu_dwdx = wu_dwdx / float(Nx)
      vu_dvdx = vu_dvdx / float(Nx)
      vv_dvdy = vv_dvdy / float(Nx)
      uv_dudy = uv_dudy / float(Nx)
      wv_dwdy = wv_dwdy / float(Nx)

      ! Don't communicate, since they're already all on rankZ = 0!

    end if

  end if


end




!----*|--.---------.---------.---------.---------.---------.---------.-|-------|
subroutine compute_Vorticity(movie)
  !----*|--.---------.---------.---------.---------.---------.---------.-|-------|
  ! Compute the RMS Vorticity and also Movie Slices of Vorticity
  ! Assumes PP Ui AND also FF CUi are stored in cr1(), etc

  character(len=35) fname
  character(len=20) gname
  logical movie
  integer i, j, k
  real(rkind) varxy(0:Nxm1, 1:Nyp), varzy(0:Nzp - 1, 1:Nyp), varxz(0:Nxm1, 0:Nzp - 1)



  !!! RMS Vorticity !!!
  ! X-component in FF space
  do j = 1, Nyp
    do k = 0, twoNkz
      do i = 0, Nxp - 1 !Nkx
        cs1(i, k, j) = cikz(k) * 0.5d0 * (cr2(i, k, j + 1) + cr2(i, k, j)) &
                       - (cr3(i, k, j + 1) - cr3(i, k, j - 1)) / (2.d0 * dyf(j))
      end do
    end do
  end do
  call fft_xz_to_physical(cs1, s1)
  ! RMS value
  do j = 1, Nyp
    omega_x(j) = 0.d0
    do k = 0, Nzp - 1
      do i = 0, Nxm1
        s1(i, k, j) = s1(i, k, j) - (1.d0 / (Ro_inv / delta)) * dTHdX(1)  ! Include the TWS!
        omega_x(j) = omega_x(j) + s1(i, k, j)**2.d0
      end do
    end do
  end do
  call mpi_allreduce(mpi_in_place, omega_x, Nyp + 2, mpi_double_precision, &
                     mpi_sum, mpi_comm_z, ierror)

  omega_x = sqrt(omega_x / float(Nx * Nz))



  ! Write Movie Slices for omega_x
  if (movie) then

    fname = 'movie.h5'
    call mpi_barrier(mpi_comm_world, ierror)
    if (rankZ == rankzmovie) then
      do j = 1, Nyp
        do i = 0, Nxm1
          varxy(i, j) = s1(i, NzMovie, j)
        end do
      end do
      write (gname,'("omegaX_xz")')
      call WriteHDF5_XYplane(fname, gname, varxy)
    end if

    if (rankY == rankymovie) then
      do j = 0, Nzp - 1
        do i = 0, Nxm1
          varxz(i, j) = s1(i, j, NyMovie)
        end do
      end do
      write (gname,'("omegaX_xy")')
      call WriteHDF5_XZplane(fname, gname, varxz)
    end if

    do j = 1, Nyp
      do i = 0, Nzp - 1
        varzy(i, j) = s1(NxMovie, i, j)
      end do
    end do
    write (gname,'("omegaX_yz")')
    call WriteHDF5_ZYplane(fname, gname, varzy)
  end if



  ! Y-component in FF space
  do j = 1, Nyp
    do k = 0, twoNkz 
      do i = 0, Nxp - 1 !Nkx
        cs1(i, k, j) = cikx(i) * cr3(i, k, j) - cikz(k) * cr1(i, k, j)
      end do
    end do
  end do
  call fft_xz_to_physical(cs1, s1)
  ! RMS value
  do j = 1, Nyp
    omega_y(j) = 0.d0
    do k = 0, Nzp - 1
      do i = 0, Nxm1
        omega_y(j) = omega_y(j) + s1(i, k, j)**2.d0
      end do
    end do
  end do
  call mpi_allreduce(mpi_in_place, omega_y, Nyp + 2, mpi_double_precision, &
                     mpi_sum, mpi_comm_z, ierror)

  omega_y = sqrt(omega_y / float(Nx * Nz))



  ! Write Movie Slices for omega_y
  if (movie) then

    fname = 'movie.h5'
    call mpi_barrier(mpi_comm_world, ierror)
    if (rankZ == rankzmovie) then
      do j = 1, Nyp
        do i = 0, Nxm1
          varxy(i, j) = s1(i, NzMovie, j)
        end do
      end do
      write (gname,'("omegaZ_xz")')
      call WriteHDF5_XYplane(fname, gname, varxy)
    end if

    if (rankY == rankymovie) then
      do j = 0, Nzp - 1
        do i = 0, Nxm1
          varxz(i, j) = s1(i, j, NyMovie)
        end do
      end do
      write (gname,'("omegaZ_xy")')
      call WriteHDF5_XZplane(fname, gname, varxz)
    end if

    do j = 1, Nyp
      do i = 0, Nzp - 1
        varzy(i, j) = s1(NxMovie, i, j)
      end do
    end do
    write (gname,'("omegaZ_yz")')
    call WriteHDF5_ZYplane(fname, gname, varzy)
  end if




  ! Z-component in FF space
  do j = 1, Nyp
    do k = 0, twoNkz
      do i = 0, Nxp - 1 ! Nkx
        cs1(i, k, j) = (cr1(i, k, j + 1) - cr1(i, k, j - 1)) / (2.d0 * dyf(j)) &
                       - cikx(i) * 0.5d0 * (cr2(i, k, j + 1) + cr2(i, k, j))
      end do
    end do
  end do
  call fft_xz_to_physical(cs1, s1)
  ! RMS value
  do j = 1, Nyp
    omega_z(j) = 0.d0
    do k = 0, Nzp - 1
      do i = 0, Nxm1
        omega_z(j) = omega_z(j) + s1(i, k, j)**2.d0
      end do
    end do
  end do
  call mpi_allreduce(mpi_in_place, omega_z, Nyp + 2, mpi_double_precision, &
                     mpi_sum, mpi_comm_z, ierror)

  omega_z = sqrt(omega_z / float(Nx * Nz))



  ! Write Movie Slices for omega_z
  if (movie) then

    fname = 'movie.h5'
    call mpi_barrier(mpi_comm_world, ierror)
    if (rankZ == rankzmovie) then
      do j = 1, Nyp
        do i = 0, Nxm1
          varxy(i, j) = s1(i, NzMovie, j)
        end do
      end do
      write (gname,'("omegaY_xz")')
      call WriteHDF5_XYplane(fname, gname, varxy)
    end if

    if (rankY == rankymovie) then
      do j = 0, Nzp - 1
        do i = 0, Nxm1
          varxz(i, j) = s1(i, j, NyMovie)
        end do
      end do
      write (gname,'("omegaY_xy")')
      call WriteHDF5_XZplane(fname, gname, varxz)
    end if

    do j = 1, Nyp
      do i = 0, Nzp - 1
        varzy(i, j) = s1(NxMovie, i, j)
      end do
    end do
    write (gname,'("omegaY_yz")')
    call WriteHDF5_ZYplane(fname, gname, varzy)
  end if




end

!----*|--.---------.---------.---------.---------.---------.---------.-|-------|
subroutine compute_azavg_and_sfluc(gname, field, flucfield)
  !----*|--.---------.---------.---------.---------.---------.---------.-|-------|
  ! Compute the azimuthal average of field then spatial fluctuation
  ! field already in Physical Space
  ! CWP 2022

  character(len=20) gname
  real(rkind), pointer, intent(in) :: field(:,:,:)
  real(rkind), pointer, intent(inout) :: flucfield(:,:,:)

  character(len=35) fname
  integer i, j, k, bin
  integer, parameter :: Nbin = Nx/2
  real(rkind) field_binned(0:Nbin-1, 0:Nyp-1)
  real(rkind) counts(0:Nbin-1, 0:Nyp-1)

  ! Sum field by radius bin
  field_binned = 0.d0
  counts = 0.d0

  do j = 1, Nyp 
    do k = 0, Nzp-1
      do i = 0, Nxm1
        ! Compute bin index
        bin = int(sqrt((gxf(i)-Lx/2.d0)**2.d0 + (gzf(rankZ*Nzp+k)-Lz/2.d0)**2.d0) * Nx/Lx)

        if (bin < Nbin) then
          ! Add values to corresponding bin and increase count (want j=0 to be bottom of domain in output)
          counts(bin, j-1) = counts(bin, j-1) + 1.d0
          field_binned(bin, j-1) = field_binned(bin, j-1) + field(i,k,j)
        end if
      end do
    end do
  end do

  ! Collect results from all processors
  call mpi_allreduce(mpi_in_place, field_binned, Nbin * Nyp, mpi_double_precision, &
                     mpi_sum, mpi_comm_z, ierror)
  call mpi_allreduce(mpi_in_place, counts, Nbin * Nyp,  mpi_double_precision, &
                     mpi_sum, mpi_comm_z, ierror)

  ! Divide by counts to get average
  do j = 0, Nyp-1
    do i = 0, Nbin-1
      if (counts(i,j) > 0) then
        field_binned(i, j) = field_binned(i, j) / counts(i, j)
      end if
    end do
  end do

  call mpi_barrier(mpi_comm_z, ierror)

  ! Write to file 
  fname = 'az_stats.h5'
  if (rankZ == 0) then
    call WriteHDF5_RYplane(fname, gname, field_binned)
  end if


  ! Compute spatial fluctuation (difference between az avg and data)
  do j = 1, Nyp 
    do k = 0, Nzp-1
      do i = 0, Nxm1
        ! Compute bin index
        bin = int(sqrt((gxf(i)-Lx/2.d0)**2.d0 + (gzf(rankZ*Nzp+k)-Lz/2.d0)**2.d0) * Nx/Lx)

        if (bin < Nbin) then
          flucfield(i, k, j) = field(i, k, j) - field_binned(bin, j-1) ! offset due to output starting at j=0, not 1
        else
          flucfield(i, k, j) = 0.d0
        end if
      end do
    end do
  end do

  ! Ensure entire fluctuation field is computed
  call mpi_barrier(mpi_comm_z, ierror)
end

!----*|--.---------.---------.---------.---------.---------.---------.-|-------|
subroutine compute_azavg(gname, field)
  !----*|--.---------.---------.---------.---------.---------.---------.-|-------|
  ! Compute the azimuthal average of field
  ! field already in Physical Space
  ! CWP 2022

  character(len=20) gname
  real(rkind), pointer, intent(in) :: field(:,:,:)

  character(len=35) fname
  integer i, j, k, bin
  integer, parameter :: Nbin = Nx/2
  real(rkind) field_binned(0:Nbin-1, 0:Nyp-1)
  real(rkind) counts(0:Nbin-1, 0:Nyp-1)

  ! Sum field by radius bin
  field_binned = 0.d0
  counts = 0.d0

  do j = 1, Nyp 
    do k = 0, Nzp-1
      do i = 0, Nxm1
        ! Compute bin index
        bin = int(sqrt((gxf(i)-Lx/2.d0)**2.d0 + (gzf(rankZ*Nzp+k)-Lz/2.d0)**2.d0) * Nx/Lx)
        
        if (bin < Nbin) then
          ! Add values to corresponding bin and increase count (want j=0 to be bottom of domain in output)
          counts(bin, j-1) = counts(bin, j-1) + 1.d0
          field_binned(bin, j-1) = field_binned(bin, j-1) + field(i,k,j)
        end if
      end do
    end do
  end do

  ! Collect results from all processors
  call mpi_allreduce(mpi_in_place, field_binned, Nbin * Nyp, mpi_double_precision, &
                     mpi_sum, mpi_comm_z, ierror)
  call mpi_allreduce(mpi_in_place, counts, Nbin * Nyp,  mpi_double_precision, &
                     mpi_sum, mpi_comm_z, ierror)

  ! Divide by counts to get average
  do j = 0, Nyp-1
    do i = 0, Nbin-1
      if (counts(i,j) > 0) then
        field_binned(i, j) = field_binned(i, j) / counts(i, j)
      end if
    end do
  end do

  call mpi_barrier(mpi_comm_z, ierror)

  ! Write to file
  fname = 'az_stats.h5'
  if (rankZ == 0) then
    call WriteHDF5_RYplane(fname, gname, field_binned)
  end if

end

!----*|--.---------.---------.---------.---------.---------.---------.-|-------|
subroutine compute_BPE
  !----*|--.---------.---------.---------.---------.---------.---------.-|-------|
  ! Compute the Background Potential Energy (BPE) -- Tseng & Ferziger 2001
  ! th(:,:,:,1) already in Physical Space
  ! Uses s1 as storage
  ! AFW 2020

  character(len=20) gname
  character(len=35) fname
  integer i, j, k, bin
  integer, parameter :: Nbin = Nx ! Useful for writing in parallel (vs 16*Ny)
  real(rkind) thmin, thmax, dTH, BPE
  real(rkind) PDF(0:Nbin - 1)
  real(rkind) Y_r(0:Nbin)
  real(rkind) th_bin(0:Nbin)
  real(rkind) DiagX(0:int(Nbin/NprocZ) - 1)

  ! Add background buoyancy gradient, store in s1
  !   Offset by Lx/2 to match the calculation of thv_m (Mean Buoyancy Production)
  s1 = 0. ! Need to clear the edges otherwise min/max value catches it!
  do j = 1, Nyp
    do k = 0, Nzp - 1
      do i = 0, Nxm1
        s1(i, k, j) = th(i, k, j, 1) + dTHdX(1) * (gx(i) - 0.5*Lx)
      end do
    end do
  end do


  if (homogeneousX) then ! Infinite Front, or the likes -- Use constant th bins

    ! Bounds of theta
    thmin = -dTHdX(1) * 0.5*Lx * 1.1
    thmax =  dTHdX(1) * 0.5*Lx * 1.1

    dTH = (thmax - thmin) / Nbin + 1.d-14

    do i = 0, Nbin
      th_bin(i) = thmin + i * dTH
    end do

    ! Compile the PDF in b
    PDF = 0.d0
    do j = jstart_th(1), jend_th(1) ! Avoid repeats at the vertical proc boundaries
      do k = 0, Nzp - 1
        do i = 0, Nxm1
          ! Compute bindex
          bin = min(int((s1(i, k, j) - thmin) / dTH), Nbin - 1) ! Catch case when bin = Nbin...

          PDF(bin) = PDF(bin) + dyf(j)
        end do
      end do
    end do

  else ! Finite Front -- Use adaptive th bins with dTH ~ sech2(z/delta)

    ! Bounds of theta
    thmin = -Ro_inv
    thmax = +Ro_inv

    do i = 0, Nbin
      th_bin(i) = -(Ro_inv + 1.d-14) * cos(i * pi / Nbin)  ! (Ro_inv + 1.d-14) * tanh( (dble(i)/Nbin - 0.5d0) * Lx/(2.d0*delta) )
    end do

    ! Compile the PDF in b
    PDF = 0.d0
    do j = jstart_th(1), jend_th(1) ! Avoid repeats at the vertical proc boundaries
      do k = 0, Nzp - 1
        do i = 0, Nxm1
          ! Do inverse transform to get bindex
          bin = min(int( acos( -min(max(s1(i, k, j),thmin),thmax) / (Ro_inv + 1.d-14) ) * Nbin / pi ), Nbin - 1) ! Catch case when bin = Nbin...

          PDF(bin) = PDF(bin) + dyf(j)
        end do
      end do
    end do


  endif



  call mpi_allreduce(mpi_in_place, PDF, Nbin, mpi_double_precision, &
                     mpi_sum, mpi_comm_world, ierror)

  ! Enforce \int_B PDF dB = 1 exactly (small dyf/2 at BCs...)  vs. /(Ly * Nx * Nz)
  PDF = PDF / sum(PDF)

  ! Compute Y_r (corresponding to TH bin edges)
  Y_r(0) = 0.d0
  do i = 1, Nbin
    ! NOTE: PDF is the distribution function, NOT density (i.e. NOT divided by dTH, so don't need another dTH!)
    Y_r(i) = Y_r(i - 1) + PDF(i - 1) ! * (th_bin(i) - th_bin(i - 1))
  end do
  Y_r = Y_r * Ly

  ! Compute BPE
  BPE = 0.d0
  do i = 0, Nbin - 1  ! Integrate
    BPE = BPE - (0.5 * (th_bin(i + 1) + th_bin(i)) * 0.5 * (Y_r(i + 1) + Y_r(i))) * &
                    (Y_r(i + 1) - Y_r(i)) / Ly
  end do


  fname = 'mean.h5'
  gname = 'BPE'
  call WriteHDF5_real(fname, gname, BPE)

  ! Write out the entire PDF and extents to construct bins
  if (rankY == 0) then
    DiagX = PDF(rankZ * int(Nbin/NprocZ):(rankZ+1) * int(Nbin/NprocZ) - 1)
    gname = 'th1PDF'
    call WriteStatH5_X(fname, gname, DiagX, int(Nbin/NprocZ))

    DiagX = th_bin(rankZ * int(Nbin/NprocZ):(rankZ+1) * int(Nbin/NprocZ) - 1)
    gname = 'th1bin'
    call WriteStatH5_X(fname, gname, DiagX, int(Nbin/NprocZ))
  end if

end

!----*|--.---------.---------.---------.---------.---------.---------.-|-------|
subroutine Compute_PDF_SVD_and_Write(gname, field, ref_field, bins, con_field, con_thresh, zstart, zstop)
  !----*|--.---------.---------.---------.---------.---------.---------.-|-------|
  ! Compute PDF of ref_field using field as weights, conditioning on con_field > con_thresh ,and normalise
  ! CWP 2023

  character(len=20) gname
  real(rkind), pointer, intent(in) :: field(:,:,:)
  real(rkind), pointer, intent(in) :: ref_field(:,:,:)
  real(rkind), pointer, intent(in) :: con_field(:,:,:)
  real(rkind), intent(in) :: bins(:)
  real(rkind) zstart, zstop, con_thresh
  
  character(len=35) fname
  integer i, j, k, l, bin
  real(rkind) field_binned(0:size(bins)-1)
  real(rkind) total_weight
  real(rkind) DiagX(0:int(size(field_binned)/NprocZ) - 1)

  field_binned = 0.d0
  total_weight = 0.d0

  do j = jstart_th(1), jend_th(1)
    do k = 0, Nzp - 1
      do i = 0, Nxm1
        if (((gyf(j) <= zstop).and.(gyf(j) >= zstart)).and.(con_field(i, k, j) > con_thresh)) then
          ! Compute bin index
          bin = -1
          do l = 1, size(bins)-1
            if ((ref_field(i, k, j) >= bins(l)) .and. (ref_field(i, k, j) < bins(l+1))) then
              bin = l-1
            end if
          end do
        
          ! Add to binned field array
          if (bin >= 0) then  ! if bin < 0 then something went awry...
            field_binned(bin) = field_binned(bin) + field(i, k, j) 
            total_weight = total_weight + field(i, k, j) * (bins(bin+2) - bins(bin+1))  ! bins is indexed from 1!
          end if

        end if 
      end do
    end do
  end do

  call mpi_allreduce(mpi_in_place, field_binned, size(bins), mpi_double_precision, &
                     mpi_sum, mpi_comm_world, ierror)
  call mpi_allreduce(mpi_in_place, total_weight, 1, mpi_double_precision, &
                     mpi_sum, mpi_comm_world, ierror)

  if (total_weight > 0.d0) then !if total_weight = 0 then field_binned = 0 too
    field_binned = field_binned / total_weight 
  end if

  fname = 'mean.h5'
  ! Write out the binned field to file
  if (rankY == 0) then
    DiagX = field_binned(rankZ * int(size(bins)/NprocZ):(rankZ+1) * int(size(bins)/NprocZ) - 1)
    call WriteStatH5_X(fname, gname, DiagX, int(size(bins)/NprocZ))
  end if

  ! Write out total weight (for un-normalising)
  gname = trim(gname)//'_w'
  call WriteHDF5_real(fname, gname, total_weight)

end

!----*|--.---------.---------.---------.---------.---------.---------.-|-------|
subroutine Compute_PDF_and_Write(gname, field, ref_field, bins, zstart, zstop)
  !----*|--.---------.---------.---------.---------.---------.---------.-|-------|
  ! Compute PDF of ref_field using field as weights, without normalisation
  ! CWP 2022
  character(len=20) gname
  real(rkind), pointer, intent(in) :: field(:,:,:)
  real(rkind), pointer, intent(in) :: ref_field(:,:,:)
  real(rkind), intent(in) :: bins(:)
  real(rkind) zstart, zstop
  
  character(len=35) fname
  integer i, j, k, l, bin
  real (rkind) field_binned(0:size(bins)-1)
  real(rkind) DiagX(0:int(size(field_binned)/NprocZ) - 1)

  field_binned = 0.d0

  do j = jstart_th(1), jend_th(1)
    do k = 0, Nzp - 1
      do i = 0, Nxm1
        if ((gyf(j) <= zstop).and.(gyf(j) >= zstart)) then
          ! Compute bin index
          bin = -1
          do l = 1, size(bins)-1
            if ((ref_field(i, k, j) > bins(l)) .and. (ref_field(i, k, j) <= bins(l+1))) then
              bin = l-1
            end if
          end do
        
          ! Add to binned field array
          if (bin >= 0) then  ! if bin < 0 then something went awry...
            field_binned(bin) = field_binned(bin) + field(i, k, j) 
          end if

        end if 
      end do
    end do
  end do

  call mpi_allreduce(mpi_in_place, field_binned, size(bins), mpi_double_precision, &
                     mpi_sum, mpi_comm_world, ierror)

  fname = 'mean.h5'
  ! Write out the binned field to file
  if (rankY == 0) then
    DiagX = field_binned(rankZ * int(size(bins)/NprocZ):(rankZ+1) * int(size(bins)/NprocZ) - 1)
    call WriteStatH5_X(fname, gname, DiagX, int(size(bins)/NprocZ))

    if (write_bins_flag) then
      gname = trim(gname)//'_bins'
      DiagX = bins(1+rankZ * int(size(bins)/NprocZ):(rankZ+1) * int(size(bins)/NprocZ) )
      call WriteStatH5_X(fname, gname, DiagX, int(size(bins)/NprocZ))
    end if
  end if

end

!----*|--.---------.---------.---------.---------.---------.---------.-|-------|
subroutine Bin_Ystar_and_Write(gname, field)
  !----*|--.---------.---------.---------.---------.---------.---------.-|-------|
  ! Bin field into z* Coordinates (Winters et al 1996)
  !   using Parallel PDF Sorting of Tseng & Ferziger 2001
  ! th(:,:,:,1) already in Physical Space
  ! field is to be binned
  ! Uses s1 as storage
  ! AFW 2021

  character(len=20) gname
  real(rkind), intent(in) :: field(:,:,:)

  character(len=35) fname
  integer i, j, k, bin
  integer, parameter :: Nbin = Nx ! Useful for writing in parallel (vs 16*Ny)
  real(rkind) thmin, thmax, dTH
  real(rkind) PDF(0:Nbin - 1)
  real(rkind) field_binned(0:Nbin - 1) ! For computing mean (field) on each z* surface
  real(rkind) DiagX(0:int(Nbin/NprocZ) - 1)

  ! Add background buoyancy gradient, store in s1
  !   Offset by Lx/2 to match the calculation of thv_m (Mean Buoyancy Production)
  s1 = 0. ! Need to clear the edges otherwise min/max value catches it!
  do j = 1, Nyp
    do k = 0, Nzp - 1
      do i = 0, Nxm1
        s1(i, k, j) = th(i, k, j, 1) + dTHdX(1) * (gx(i) - 0.5*Lx)
      end do
    end do
  end do


  if (homogeneousX) then ! Infinite Front, or the likes -- Use constant th bins

    ! Bounds of theta
    thmin = -dTHdX(1) * 0.5*Lx * 1.1
    thmax =  dTHdX(1) * 0.5*Lx * 1.1

    dTH = (thmax - thmin) / Nbin + 1.d-14

    ! Compile the PDF in b
    PDF = 0.d0
    field_binned = 0.d0
    do j = jstart_th(1), jend_th(1) ! Avoid repeats at the vertical proc boundaries
      do k = 0, Nzp - 1
        do i = 0, Nxm1
          ! Compute bindex
          bin = min(int((s1(i, k, j) - thmin) / dTH), Nbin - 1) ! Catch case when bin = Nbin...

          PDF(bin) = PDF(bin) + dyf(j)
          field_binned(bin) = field_binned(bin) + field(i, k, j) * dyf(j)
        end do
      end do
    end do

  else ! Finite Front -- Use adaptive th bins with dTH ~ sech2(z/delta)

    ! Bounds of theta
    thmin = -Ro_inv
    thmax = +Ro_inv

    ! Compile the PDF in b
    PDF = 0.d0
    field_binned = 0.d0
    do j = jstart_th(1), jend_th(1) ! Avoid repeats at the vertical proc boundaries
      do k = 0, Nzp - 1
        do i = 0, Nxm1
          ! Do inverse transform to get bindex
          bin = min(int( acos( -min(max(s1(i, k, j),thmin),thmax) / (Ro_inv + 1.d-14) ) * Nbin / pi ), Nbin - 1) ! Catch case when bin = Nbin...

          PDF(bin) = PDF(bin) + dyf(j)
          field_binned(bin) = field_binned(bin) + field(i, k, j) * dyf(j)
        end do
      end do
    end do

  endif


  call mpi_allreduce(mpi_in_place, PDF, Nbin, mpi_double_precision, &
                     mpi_sum, mpi_comm_world, ierror)
  call mpi_allreduce(mpi_in_place, field_binned, Nbin, mpi_double_precision, &
                     mpi_sum, mpi_comm_world, ierror)

  ! Divide by sum(dyf), i.e. PDF, to get a bin-average
  field_binned = field_binned / PDF

  fname = 'mean.h5'
  ! Write out the entire PDF and extents to construct bins
  if (rankY == 0) then
    DiagX = field_binned(rankZ * int(Nbin/NprocZ):(rankZ+1) * int(Nbin/NprocZ) - 1)
    call WriteStatH5_X(fname, gname, DiagX, int(Nbin/NprocZ))
  end if

end









!----*|--.---------.---------.---------.---------.---------.---------.-|-------|
subroutine integrate_y_var(var, res)
  !----*|--.---------.---------.---------.---------.---------.---------.-|-------|
  ! Integrates a vector in Y

  integer j
  real(rkind) var(0:Nyp + 1), res

  res = 0.
  do j = 2, Nyp
    res = res + 0.5 * (var(j) + var(j - 1)) * dy(j)
  end do
  call mpi_allreduce(mpi_in_place, res, 1, &
                     mpi_double_precision, mpi_sum, mpi_comm_y, ierror)

  res = res / Ly ! To get average

end subroutine


!----*|--.---------.---------.---------.---------.---------.---------.-|-------|
subroutine integrate_z_var(var, res)
  !----*|--.---------.---------.---------.---------.---------.---------.-|-------|
  ! Integrates a full 3D cube across Z

  integer i, k, j
  real(rkind) var(0:Nx + 1, 0:Nzp + 1, 0:Nyp + 1), res(0:Nxm1, 1:Nyp)

  do i = 0, Nxm1
    do j = 1, Nyp
      res(i, j) = 0.
      do k = 0, Nzp
        res(i, j) = res(i, j) + var(i, k, j) * dz(k)
      end do
    end do
  end do
  call mpi_allreduce(mpi_in_place, res, Nx * Nyp, &
                     mpi_double_precision, mpi_sum, mpi_comm_z, ierror)
  call mpi_barrier(mpi_comm_world, ierror)

  res = res / Lz ! Gives average

end subroutine



!----*|--.---------.---------.---------.---------.---------.---------.-|-------|
subroutine reduce_and_write_XYplane(fname, gname, res, allreduce, movie)
  !----*|--.---------.---------.---------.---------.---------.---------.-|-------|
  ! Communicates X-Y plane and sums across all Z processors.
  ! Sends the result to all processors if allreduce == .true.

  character(len=35) fname
  character(len=20) gname
  logical movie
  real(rkind) res(0:Nxm1, 1:Nyp)
  logical allreduce, writeHDF5

  if (allreduce) then
    call mpi_allreduce(mpi_in_place, res, Nx * Nyp, &
                       mpi_double_precision, mpi_sum, mpi_comm_z, ierror)
  else
    ! if (.not. movie) return ! Why are we doing this otherwise?

    if (rankZ == 0) then
      call mpi_reduce(mpi_in_place, res, Nx * Nyp, &
                      mpi_double_precision, mpi_sum, 0, mpi_comm_z, ierror)
    else ! Don't use mpi_in_place for other processes, except for allreduce...
      call mpi_reduce(res, 0, Nx * Nyp, &
                      mpi_double_precision, mpi_sum, 0, mpi_comm_z, ierror)
    end if
  end if

  call mpi_barrier(mpi_comm_world, ierror)

  if (movie .and. rankZ == 0) then
    call WriteHDF5_XYplane(fname, gname, res)
  end if

end subroutine











! ---- LES ----

!----*|--.---------.---------.---------.---------.---------.---------.-|-------|
subroutine compute_TKE_diss_les
  !----*|--.---------.---------.---------.---------.---------.---------.-|-------|
  ! Calculate the componet of the SGS dissipation rate
  ! only includes the terms timestepped implicitly

  character(len=35) fname
  character(len=20) gname
  real(rkind) eps_sgs2(1:Nyp)
  real(rkind) Diag(Nyp)
  integer i, j, k

  ! Compute the turbulent dissipation rate, epsilon=nu*<du_i/dx_j du_i/dx_j>
  ! At *consistent* GY points!

  ! Compute the contribution at GYF first. Store in S1.
  do j = 1, Nyp
    do k = 0, Nzp - 1
      do i = 0, Nxm1
        s1(i, k, j) = u1(i, k, j) * &
                        ((nu_t(i, k, j + 1) * (u1(i, k, j + 1) - u1(i, k, j)) / dy(j + 1) &
                               - nu_t(i, k, j) * (u1(i, k, j) - u1(i, k, j - 1)) / dy(j)) &
                           / dyf(j)) &  ! At GYF
                     + u3(i, k, j) * &
                        ((nu_t(i, k, j + 1) * (u3(i, k, j + 1) - u3(i, k, j)) / dy(j + 1) &
                               - nu_t(i, k, j) * (u3(i, k, j) - u3(i, k, j - 1)) / dy(j)) &
                           / dyf(j))  ! At GYF
      end do
    end do
  end do

! Then, interpolate the u1 & u2 contribution onto GY
! so that it conserves the dissipation as in code
  do j = 2, Nyp
    do k = 0, Nzp - 1
      do i = 0, Nxm1
        temp(i, k, j) = 0.5d0 * (s1(i, k, j) + s1(i, k, j - 1)) & ! u1 & u3 at GY
                      + u2(i, k, j) * &
                        ((0.5d0 * (nu_t(i, k, j) + nu_t(i, k, j + 1)) * (u2(i, k, j + 1) - u2(i, k, j)) &
                            / dyf(j) &
                        - 0.5d0 * (nu_t(i, k, j) + nu_t(i, k, j - 1)) * (u2(i, k, j) - u2(i, k, j - 1)) &
                            / dyf(j - 1)) / dy(j)) ! At GY
      end do
    end do
  end do

  ! Now calculate the horizontal average
  do j = 1, Nyp
    eps_sgs2(j) = 0.d0
    do i = 0, Nxm1
      do k = 0, Nzp - 1
        eps_sgs2(j) = eps_sgs2(j) + temp(i, k, j)
      end do
    end do
  end do

  call mpi_allreduce(mpi_in_place, eps_sgs2, Nyp &
                     , mpi_double_precision, &
                     mpi_sum, mpi_comm_z, ierror)

  fname = 'mean.h5'

  if (rankZ == 0) then
    gname = 'eps_sgs2'
    Diag = eps_sgs2(1:Nyp) / float(Nx * Nz)
    call WriteStatH5_Y(fname, gname, Diag)
  end if

end


!----*|--.---------.---------.---------.---------.---------.---------.-|-------|
subroutine save_stats_LES_OOL(blank)
  !----*|--.---------.---------.---------.---------.---------.---------.-|-------|

  integer n
  character(len=35) fname
  character(len=20) gname
  logical blank
  real(rkind) :: Diag(1:Nyp)

  ! Store/write 2D slices
  real(rkind) varxy(0:Nxm1, 1:Nyp), varzy(0:Nzp - 1, 1:Nyp), varxz(0:Nxm1, 0:Nzp - 1)


  if (blank) then
    fname = 'mean.h5'

    if (rankZ == 0) then
      Diag = 0.d0
      gname = 'nu_sgs'
      call WriteStatH5_Y(fname, gname, Diag)

      gname = 'eps_sgs1'
      call WriteStatH5_Y(fname, gname, Diag)

      Diag = 0.d0
      gname = 'kappa_sgs'
      call WriteStatH5_Y(fname, gname, Diag)
    end if

    do n = 1, N_th
      fname = 'movie.h5'
      if (rankZ == rankzmovie) then
        varxy = 0.d0
        write (gname,'("kappa_t", I0.1 "_xz")') n
        call WriteHDF5_XYplane(fname, gname, varxy)
      end if

      if (rankY == rankymovie) then
        varxz = 0.d0
        write (gname,'("kappa_t", I0.1 "_xy")') n
        call WriteHDF5_XZplane(fname, gname, varxz)
      end if
    
      varzy = 0.d0
      write (gname,'("kappa_t", I0.1 "_yz")') n
      call WriteHDF5_ZYplane(fname, gname, varzy)
    end do

  else
    ! Needed to write out LES Statistics without timestepping...
    ! DON'T run this except for when stopping the simulation!

    rk_step = 1
    flag_save_LES = .true.

    call les_chan
    call les_chan_th

    call fft_xz_to_fourier(u1, cu1)
    call fft_xz_to_fourier(u2, cu2)
    call fft_xz_to_fourier(u3, cu3)

    do n = 1, N_th
      call fft_xz_to_fourier(th(:, :, :, n), cth(:, :, :, n))
    end do

  end if

  return

end
