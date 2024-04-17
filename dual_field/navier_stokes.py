import firedrake
import ufl

import dual_field

class NSPrimal:
    #: int: The polynomial degree to use in the approximating spaces, following a de Rham sequence
    p = 1

    #: bool: Specified if the domain is periodic or not
    periodic = True

    # Temporal parameters
    #: double: The start time instant
    t = 0.0
    #: double: The time step size
    dt = 0.1

    #: double: The inverse of the fluid Reynolds number
    inv_Rf = 1.0

    #: string: The run ID used the name of the folder where to store the output files
    run_id = 'default_primal_id'

    #: firedrake.mesh.MeshGeometry: The mesh to use to solve the problem
    mesh = None

    # Function spaces
    #: firedrake.functionspaceimpl.WithGeometry: The H(curl) discrete function space
    U_space = None
    #: firedrake.functionspaceimpl.WithGeometry: The H(div) discrete function space
    W_space = None
    #: firedrake.functionspaceimpl.WithGeometry: The H(grad) discrete function space
    P_space = None
    #: firedrake.functionspaceimpl.WithGeometry: The R discrete function space Lagrange multiplier for Pressure
    R_space = None

    # Mixed function space
    #: firedrake.functionspaceimpl.WithGeometry: The mixed space for the primal system to solve for
    #                                            velocity, vorticity and pressure.
    mixed_space = None

    # Null spaces
    #: firedrake.functionspaceimpl.WithGeometry: The null space for pressure.
    p_nullspace = None
    #: firedrake.functionspaceimpl.WithGeometry: The mixed space where to search for solutions
    #                                            in the null space (constant pressures).
    mixed_nullspace = None

    # Basis functions
    # Trial functions
    epsilon1_trial = None
    epsilon2_trial = None
    epsilon0_trial = None
    # Test functions
    epsilon1_test = None
    epsilon2_test = None
    epsilon0_test = None

    # Solution functions

    # Mixed system functions with solutions at different time instants
    #: firedrake.functionspaceimpl.WithGeometry: Mixed space for initial step: velocity, vorticity, pressure
    mixed_variables_0 = None
    #: firedrake.functionspaceimpl.WithGeometry: Mixed space for next step: velocity, vorticity, pressure
    mixed_variables_1 = None

    # Individual functions with solutions at different time instants
    #: firedrake.functionspaceimpl.WithGeometry: Primal pressure at time step :math:`k + \\frac{1}{2}`
    p_1_2 = None
    #: firedrake.functionspaceimpl.WithGeometry: Primal velocity at time step :math:`k`
    u_0 = None
    #: firedrake.functionspaceimpl.WithGeometry: Primal velocity at time step :math:`k + 1`
    u_1 = None
    #: firedrake.functionspaceimpl.WithGeometry: Primal vorticity at time step :math:`k`
    w_0 = None
    #: firedrake.functionspaceimpl.WithGeometry: Primal velocity at time step :math:`k + 1`
    w_1 = None

    # Weak forms
    #: ufl.form.Form: Bilinear form associated to the time-constant parts of the primal momentum, vorticity,
    #                 and continuity equations (left-hand side).
    A1 = None
    #: ufl.form.Form: Bilinear form associated to the time-variable part of the primal momentum, vorticity,
    #                 and continuity equations (left-hand side) only the momentum equation part is non-zero,
    #                 all other equations do not have time evolution terms.
    A2 = None
    #: ufl.form.Form: The bilinear form associated to the time-constant parts of the primal momentum, vorticity,
    # and continuity equations (left-hand side).
    b = None

    # Parameters for linear algebra solver
    solver_parameters = None

    def __init__(self, mesh: firedrake.mesh.MeshGeometry, dt=0.1, inv_Rf=1.0, p=1):
        print('Initializing ns_primal...')

        self.dt = dt
        self.inv_Rf = inv_Rf 
        self.p = p

        self.mesh = mesh

        self.__init_solver_parameters()

        self.__init_function_spaces()


    def init_weak_forms(self, w_dual_1_2, f_expression=firedrake.as_vector([firedrake.Constant(0.0), firedrake.Constant(0.0), firedrake.Constant(0.0)])):
        # Bilinear variational forms
        # The bilinear form associated to the time-constant parts of the primal momentum, vorticity,
        # and continuity equations (left-hand side)
        self.A1 = firedrake.inner(self.epsilon1_trial, self.epsilon1_test) * firedrake.dx + \
            self.dt * firedrake.inner(firedrake.grad(self.epsilon0_trial), self.epsilon1_test) * firedrake.dx + \
            0.5 * self.dt * self.inv_Rf * firedrake.inner(self.epsilon2_trial, firedrake.curl(self.epsilon1_test)) * firedrake.dx + \
            firedrake.inner(self.epsilon2_trial, self.epsilon2_test) * firedrake.dx - \
            firedrake.inner(firedrake.curl(self.epsilon1_trial), self.epsilon2_test) * firedrake.dx + \
            firedrake.inner(self.epsilon1_trial, firedrake.grad(self.epsilon0_test)) * firedrake.dx

        # The bilinear form associated to the time-variable part of the primal momentum, vorticity,
        # and continuity equations (left-hand side) only the momentum equation part is non-zero,
        # all other equations do not have time evolution terms.
        self.A2 = -0.5 * self.dt * firedrake.inner(dual_field.cross(self.epsilon1_trial, w_dual_1_2), self.epsilon1_test) * firedrake.dx

        # The linear form associated to the right-hand side of the primal system.
        self.b = firedrake.inner(self.u_0, self.epsilon1_test) * firedrake.dx - \
            0.5 * self.dt * self.inv_Rf * firedrake.inner(self.w_0, firedrake.curl(self.epsilon1_test)) * firedrake.dx + \
            0.5 * self.dt * firedrake.inner(dual_field.cross(self.u_0, w_dual_1_2), self.epsilon1_test) * firedrake.dx + \
            self.dt * firedrake.inner(f_expression, self.epsilon1_test) * firedrake.dx

    def set_initial_conditions(self, velocity_input, vorticity_input):
        if self.mesh.geometric_dimension() == 3:
            if (self.mesh.ufl_cell() == ufl.tetrahedron):
                self.u_0.interpolate(velocity_input)  # set the initial condition t = 0 for velocity field time step k = 0
                self.w_0.interpolate(vorticity_input)  # set initial condition t= 0 for velocity field time step k = 0
            else:
                self.u_0.project(velocity_input)  # set the initial condition t = 0 for velocity field time step k = 0
                self.w_0.project(vorticity_input)  # set initial condition t= 0 for velocity field time step k = 0    
        else:
            if (self.mesh.ufl_cell() == ufl.triangle):
                self.u_0.interpolate(velocity_input)  # set the initial condition t = 0 for velocity field time step k = 0
                self.w_0.interpolate(vorticity_input)  # set initial condition t= 0 for velocity field time step k = 0
            else:
                self.u_0.project(velocity_input)  # set the initial condition t = 0 for velocity field time step k = 0
                self.w_0.project(vorticity_input)  # set initial condition t= 0 for velocity field time step k = 0  

        self.u_1.assign(self.u_0)
        self.w_1.assign(self.w_0)

    def time_step(self):
        # Assemble the system
        # For now, the whole system is assembled, I tried to assemble the time independent part
        # and then only the time-dependent part every time step, but I was not able to sum them,
        # so now all are assembled, which I think is slightly less efficient
        A = firedrake.assemble(self.A1 + self.A2)
        B = firedrake.assemble(self.b)

        # Solve
        firedrake.solve(A, self.mixed_variables_1, B,
                        solver_parameters=self.solver_parameters,
                        nullspace=self.mixed_nullspace)

        # Restart the variables for time stepping again
        self.mixed_variables_0.assign(self.mixed_variables_1)

        # Update time 
        self.t += self.dt

    def error_velocity(self, velocity_reference):
        # Computes the error of the computed velocity field with respect to velocity_reference
        # velocity_reference must be an expression ready to be evaluated, i.e., it must correspond
        # to the current time instant (integer time instant t = k * self.dt)
        return firedrake.errornorm(velocity_reference, self.u_0, norm_type='L2')
    
    def error_vorticity(self, vorticity_reference):
        # Computes the error of the computed vorticity field with respect to vorticity_reference
        # vorticity_reference must be an expression ready to be evaluated, i.e., it must correspond
        # to the current time instant (integer time instant t = k * self.dt)
        return firedrake.errornorm(vorticity_reference, self.w_0, norm_type='L2')
    
    def error_pressure(self, pressure_reference):
        # Computes the error of the computed pressure field with respect to pressure_reference
        # pressure_reference must be an expression ready to be evaluated, i.e., it must correspond
        # to the current time instant (fractional time instant t = (k - 1/2) * self.dt)
        return firedrake.errornorm(pressure_reference, self.p_1_2, norm_type='L2')

    def __init_function_spaces(self):
        # Individual function spaces
        if self.mesh.geometric_dimension() == 3:
            if (self.mesh.ufl_cell() == ufl.tetrahedron):
                # For tetrahedral meshes
                self.U_space = firedrake.FunctionSpace(self.mesh, "N1curl", self.p)  # the H(curl) space
                self.W_space = firedrake.FunctionSpace(self.mesh, "RT", self.p)  # the H(div) space
                self.P_space = firedrake.FunctionSpace(self.mesh, "CG", self.p)  # the H(grad) space
                self.R_space = firedrake.FunctionSpace(self.mesh, 'R', 0)  # Lagrangian space for pressure to have integral 1
            else:
                raise TypeError("3D hexahedral meshes not available yet!")
            
        else:
            if (self.mesh.ufl_cell() == ufl.triangle):
                # For tetrahedral meshes
                self.U_space = firedrake.FunctionSpace(self.mesh, "N1curl", self.p)  # the H(curl) space
                self.W_space = firedrake.FunctionSpace(self.mesh, "DG", self.p - 1)  # the H(div) space
                self.P_space = firedrake.FunctionSpace(self.mesh, "CG", self.p)  # the H(grad) space
                self.R_space = firedrake.FunctionSpace(self.mesh, 'R', 0)  # Lagrangian space for pressure to have integral 1
            else:
                # For quadrilateral meshes
                self.U_space = firedrake.FunctionSpace(self.mesh, "RTCE", self.p)  # the H(curl) space
                self.W_space = firedrake.FunctionSpace(self.mesh, "DG", self.p - 1)  # the H(div) space
                self.P_space = firedrake.FunctionSpace(self.mesh, "CG", self.p)  # the H(grad) space
                self.R_space = firedrake.FunctionSpace(self.mesh, 'R', 0)  # Lagrangian space for pressure to have integral 1
        

        # self.G_space = firedrake.FunctionSpace(self.mesh, "CG", self.p)  # the H(grad) space
        # self.C_space = firedrake.FunctionSpace(self.mesh, "N1curl", self.p)  # the H(curl) space
        # self.D_space = firedrake.FunctionSpace(self.mesh, "RT", self.p)  # the H(div) space
        # self.S_space = firedrake.FunctionSpace(self.mesh, "DG", self.p - 1)  # the L2 space
        # self.R_space = firedrake.FunctionSpace(self.mesh, 'R', 0)  # Lagrangian space for pressure to have integral 1

        # Mixed function spaces
        self.mixed_space = self.U_space * self.W_space * self.P_space
        # self.mixed_space = self.C_space * self.D_space * self.G_space

        # Mixed null space to solve more efficiently for pressure
        self.p_nullspace = firedrake.VectorSpaceBasis(constant=True)
        self.mixed_nullspace = firedrake.MixedVectorSpaceBasis(self.mixed_space,
            [self.mixed_space.sub(0), self.mixed_space.sub(1), self.p_nullspace])

        # Trial and test functions
        self.epsilon1_trial, self.epsilon2_trial, self.epsilon0_trial = firedrake.TrialFunctions(self.mixed_space)
        self.epsilon1_test, self.epsilon2_test, self.epsilon0_test = firedrake.TestFunctions(self.mixed_space)

        # Solution variables
        # Mixed system functions with solutions at different time instants
        self.mixed_variables_0 = firedrake.Function(self.mixed_space)  # the mixed primal functions at initial step: velocity, vorticity, pressure
        self.mixed_variables_1 = firedrake.Function(self.mixed_space)  # the mixed primal functions at next step: velocity, vorticity, pressure

        # Individual functions with solutions at different time instants
        self.p_1_2 = self.mixed_variables_1.sub(2)  # pressure, 1_2 means half step k + 1/2

        self.u_0 = self.mixed_variables_0.sub(0)  # velocity, 0 means integer step k
        self.u_1 = self.mixed_variables_1.sub(0)  # velocity, 1 means integer step k + 1

        self.w_0 = self.mixed_variables_0.sub(1)  # vorticity, 0 means integer step k
        self.w_1 = self.mixed_variables_1.sub(1)  # vorticity, 1 means integer step k + 1

    def __init_solver_parameters(self):
        # Solver parameters
        # self.solver_parameters = {'ksp_type': 'preonly',
        #                      'pc_type': 'lu',
        #                      # 'mat_type': 'aij',
        #                      'pc_factor_mat_solver_type': 'mumps',
        #                      # 'ksp_monitor_true_residual': None,
        #                     #  'ksp_initial_guess_nonzero': True,
        #                      'ksp_converged_reason': None}
        
        self.solver_parameters = {'ksp_type': 'gmres',
                             'pc_type': 'asm',
                             # 'mat_type': 'aij',
                             'pc_factor_mat_solver_type': 'mumps',
                             # 'ksp_monitor_true_residual': None,
                             'ksp_initial_guess_nonzero': True,
                             'ksp_converged_reason': None}

        # self.solver_parameters = {"pc_type": "fieldsplit",
        #                      "pc_fieldsplit_type": "multiplicative",
        #                      # first split contains first two fields, second
        #                      # contains the third
        #                      # 'mat_type': 'aij',
        #                      "pc_fieldsplit_0_fields": "0, 2",
        #                      "pc_fieldsplit_1_fields": "1",
        #                      # Multiplicative fieldsplit for first field
        #                      "fieldsplit_0_pc_type": "fieldsplit",
        #                      "fieldsplit_0_pc_fieldsplit_type": "schur",
        #                      #  "fieldsplit_0_pc_fieldsplit_schur_fact_type": "lower",
        #                      # LU on each field
        #                      "fieldsplit_0_fieldsplit_0_pc_type": "ilu",
        #                      "fieldsplit_0_fieldsplit_1_pc_type": "jacobi",
        #                      # ILU on the schur complement block
        #                      "fieldsplit_1_pc_type": "ilu",
        #                      'ksp_initial_guess_nonzero': True,
        #                      # 'ksp_monitor_true_residual': None,
        #                      'ksp_converged_reason': None}

        # self.solver_parameters = {
        #                      "ksp_type": "fgmres",
        #                      "ksp_gmres_restart": 100,
        #                      "pc_type": "fieldsplit",
        #                      "pc_fieldsplit_type": "multiplicative",
        #                      # first split contains first two fields, second
        #                      # contains the third
        #                      # 'mat_type': 'aij',
        #                      'ksp_rtol': 1e-8,
        #                      "pc_fieldsplit_0_fields": "0, 2",
        #                      "pc_fieldsplit_1_fields": "1",
        #                      # Multiplicative fieldsplit for first field
        #                      "fieldsplit_0_pc_type": "fieldsplit",
        #                      "fieldsplit_0_pc_fieldsplit_type": "schur",
        #                      #  "fieldsplit_0_pc_fieldsplit_schur_fact_type": "lower",
        #                      # LU on each field
        #                      "fieldsplit_0_fieldsplit_0_ksp_type": "preonly",
        #                      "fieldsplit_0_fieldsplit_0_pc_type": "lu",
        #                      "fieldsplit_0_fieldsplit_1_pc_type": "jacobi",
        #                      # ILU on the schur complement block
        #                      "fieldsplit_1_ksp_type": "preonly",
        #                      "fieldsplit_1_pc_type": "lu",
        #                      'ksp_initial_guess_nonzero': True,
        #                      'ksp_monitor': None,
        #                      'ksp_converged_reason': None}
        
        # self.solver_parameters = {
        #                      "ksp_type": "fgmres",
        #                      "ksp_gmres_restart": 100,
        #                      "pc_type": "fieldsplit",
        #                      "pc_fieldsplit_type": "multiplicative",
        #                      # first split contains first two fields, second
        #                      # contains the third
        #                      # 'mat_type': 'aij',
        #                      'ksp_rtol': 1e-8,
        #                      "pc_fieldsplit_0_fields": "0, 2",
        #                      "pc_fieldsplit_1_fields": "1",
        #                      # Multiplicative fieldsplit for first field
        #                      "fieldsplit_0_pc_type": "fieldsplit",
        #                      "fieldsplit_0_pc_fieldsplit_type": "schur",
        #                      #  "fieldsplit_0_pc_fieldsplit_schur_fact_type": "lower",
        #                      # LU on each field
        #                      "fieldsplit_0_fieldsplit_0_ksp_type": "preonly",
        #                      "fieldsplit_0_fieldsplit_0_pc_type": "lu",
        #                      "fieldsplit_0_fieldsplit_1_pc_type": "jacobi",
        #                      # ILU on the schur complement block
        #                      "fieldsplit_1_ksp_type": "preonly",
        #                      "fieldsplit_1_pc_type": "lu",
        #                      'ksp_initial_guess_nonzero': True,
        #                      'ksp_monitor': None,
        #                      'ksp_converged_reason': None}

class NSDual:
    #: int: The polynomial degree to use in the approximating spaces, following a de Rham sequence
    p = 1

    #: bool: Specified if the domain is periodic or not
    periodic = True

    # Temporal parameters
    #: double: The start time instant
    t = 0.0
    #: double: The time step size
    dt = 0.1

    #: double: The inverse of the fluid Reynolds number
    inv_Rf = 1.0

    #: string: The run ID used the name of the folder where to store the output files
    run_id = 'default_dual_id'

    #: firedrake.mesh.MeshGeometry: The mesh to use to solve the problem
    mesh = None

    # Function spaces
    #: firedrake.functionspaceimpl.WithGeometry: The H(curl) discrete function space
    W_space = None
    #: firedrake.functionspaceimpl.WithGeometry: The H(div) discrete function space
    U_space = None
    #: firedrake.functionspaceimpl.WithGeometry: The L2 discrete function space
    P_space = None
    #: firedrake.functionspaceimpl.WithGeometry: The R discrete function space Lagrange multiplier for Pressure
    RSpace = None

    # Mixed function space
    #: firedrake.functionspaceimpl.WithGeometry: The mixed space for the dual system to solve for
    #                                            velocity, vorticity and pressure.
    mixed_space = None

    # Null spaces
    #: firedrake.functionspaceimpl.WithGeometry: The null space for pressure.
    p_nullspace = None
    #: firedrake.functionspaceimpl.WithGeometry: The mixed space where to search for solutions
    #                                            in the null space (constant pressures).
    mixed_nullspace = None

    # Basis functions
    # Trial functions
    epsilon2_trial = None
    epsilon1_trial = None
    epsilon3_trial = None
    # Test functions
    epsilon2_test = None
    epsilon1_test = None
    epsilon3_test = None

    # Solution functions

    # Mixed system functions with solutions at different time instants
    #: firedrake.functionspaceimpl.WithGeometry: Dual mixed space for initial step: velocity, vorticity, pressur
    mixed_variables_0 = None
    #: firedrake.functionspaceimpl.WithGeometry: Dual mixed space for next step: velocity, vorticity, pressure
    mixed_variables_1 = None

    # Individual functions with solutions at different time instants
    #: firedrake.functionspaceimpl.WithGeometry: Dual pressure at time step :math:`k + 1`
    p_1 = None
    #: firedrake.functionspaceimpl.WithGeometry: Dual velocity at time step :math:`k + \\frac{1}{2}`
    u_1_2 = None
    #: firedrake.functionspaceimpl.WithGeometry: Dual velocity at time step :math:`k + \\frac{3}{2}`
    u_3_2 = None
    #: firedrake.functionspaceimpl.WithGeometry: Dual vorticity at time step :math:`k + \\frac{1}{2}`
    w_1_2 = None
    #: firedrake.functionspaceimpl.WithGeometry: Dual vorticity at time step :math:`k + \\frac{3}{2}`
    w_3_2 = None

    # Weak forms
    #: ufl.form.Form: Bilinear form associated to the time-constant parts of the dual momentum, vorticity,
    #                 and continuity equations (left-hand side).
    A1 = None
    #: ufl.form.Form: Bilinear form associated to the time-variable part of the dual momentum, vorticity,
    #                 and continuity equations (left-hand side) only the momentum equation part is non-zero,
    #                 all other equations do not have time evolution terms.
    A2 = None
    #: ufl.form.Form: The bilinear form associated to the time-constant parts of the dual momentum, vorticity,
    # and continuity equations (left-hand side).
    b = None

    # Parameters for linear algebra solver
    solver_parameters = None

    def __init__(self, mesh: firedrake.mesh.MeshGeometry, dt=0.1, inv_Rf=1.0, p=1):
        print('Initializing ns_dual...')

        self.dt = dt
        self.inv_Rf = inv_Rf 
        self.p = p

        self.mesh = mesh

        self.__init_solver_parameters()

        self.__init_function_spaces()

    def init_weak_forms(self, w_primal_1, f_expression=firedrake.as_vector([firedrake.Constant(0.0), firedrake.Constant(0.0), firedrake.Constant(0.0)])):
        # Bilinear variational forms
        # The bilinear form associated to the time-constant parts of the dual momentum, vorticity,
        # and continuity equations (left-hand side)
        self.A1 = firedrake.inner(self.epsilon2_trial, self.epsilon2_test) * firedrake.dx - \
            self.dt * firedrake.inner(self.epsilon3_trial, firedrake.div(self.epsilon2_test)) * firedrake.dx + \
            0.5 * self.dt * self.inv_Rf * firedrake.inner(firedrake.curl(self.epsilon1_trial), self.epsilon2_test) * firedrake.dx + \
            firedrake.inner(self.epsilon1_trial, self.epsilon1_test) * firedrake.dx - \
            firedrake.inner(self.epsilon2_trial, firedrake.curl(self.epsilon1_test)) * firedrake.dx + \
            firedrake.inner(firedrake.div(self.epsilon2_trial), self.epsilon3_test) * firedrake.dx

        # The bilinear form associated to the time-variable part of the dual momentum, vorticity,
        # and continuity equations (left-hand side) only the momentum equation part is non-zero,
        # all other equations do not have time evolution terms.
        self.A2 = -0.5 * self.dt * firedrake.inner(dual_field.cross(self.epsilon2_trial, w_primal_1), self.epsilon2_test) * firedrake.dx

        # The linear form associated to the right-hand side of the dual system.
        self.b = firedrake.inner(self.u_1_2, self.epsilon2_test) * firedrake.dx - \
            0.5 * self.dt * self.inv_Rf * firedrake.inner(firedrake.curl(self.w_1_2), self.epsilon2_test) * firedrake.dx + \
            0.5 * self.dt * firedrake.inner(dual_field.cross(self.u_1_2, w_primal_1), self.epsilon2_test) * firedrake.dx + \
            self.dt * firedrake.inner(f_expression, self.epsilon2_test) * firedrake.dx

    def set_initial_conditions(self, velocity_input, vorticity_input):
        if self.mesh.geometric_dimension() == 3:
            if (self.mesh.ufl_cell() == ufl.tetrahedron):
                self.u_1_2.interpolate(velocity_input)  # set the initial condition t = 0 for velocity field time step k = 0
                self.w_1_2.interpolate(vorticity_input)  # set initial condition t= 0 for velocity field time step k = 0
            else:
                self.u_1_2.project(velocity_input)  # set the initial condition t = 0 for velocity field time step k = 0
                self.w_1_2.project(vorticity_input)  # set initial condition t= 0 for velocity field time step k = 0    
        else:
            if (self.mesh.ufl_cell() == ufl.triangle):
                self.u_1_2.interpolate(velocity_input)  # set the initial condition t = 0 for velocity field time step k = 0
                self.w_1_2.interpolate(vorticity_input)  # set initial condition t= 0 for velocity field time step k = 0
            else:
                self.u_1_2.project(velocity_input)  # set the initial condition t = 0 for velocity field time step k = 0
                self.w_1_2.project(vorticity_input)  # set initial condition t= 0 for velocity field time step k = 0 

        self.u_3_2.assign(self.u_1_2)
        self.w_3_2.assign(self.w_1_2)

    def time_step(self):
        # Assemble all at once, every time step, see self.time_step in primal for a slightly
        # longer discussion
        A = firedrake.assemble(self.A1 + self.A2)
        B = firedrake.assemble(self.b)

        # Solve
        firedrake.solve(A, self.mixed_variables_1, B,
                        solver_parameters=self.solver_parameters,
                        nullspace=self.mixed_nullspace)

        # Restart the variables for time stepping again
        self.mixed_variables_0.assign(self.mixed_variables_1)

    def error_velocity(self, velocity_reference):
        # Computes the error of the computed velocity field with respect to velocity_reference
        # velocity_reference must be an expression ready to be evaluated, i.e., it must correspond
        # to the current time instant (integer time instant t = k * self.dt)
        return firedrake.errornorm(velocity_reference, self.u_1_2, norm_type='L2')
    
    def error_vorticity(self, vorticity_reference):
        # Computes the error of the computed vorticity field with respect to vorticity_reference
        # vorticity_reference must be an expression ready to be evaluated, i.e., it must correspond
        # to the current time instant (integer time instant t = k * self.dt)
        return firedrake.errornorm(vorticity_reference, self.w_1_2, norm_type='L2')
    
    def error_pressure(self, pressure_reference):
        # Computes the error of the computed pressure field with respect to pressure_reference
        # pressure_reference must be an expression ready to be evaluated, i.e., it must correspond
        # to the current time instant (fractional time instant t = (k - 1/2) * self.dt)
        return firedrake.errornorm(pressure_reference, self.p_1, norm_type='L2')

    def __init_function_spaces(self):
        # Individual function spaces
        if self.mesh.geometric_dimension() == 3:
            if (self.mesh.ufl_cell() == ufl.tetrahedron):
                # For tetrahedral meshes
                self.U_space = firedrake.FunctionSpace(self.mesh, "RT", self.p)  # the H(curl) space
                self.W_space = firedrake.FunctionSpace(self.mesh, "N1curl", self.p)  # the H(div) space
                self.P_space = firedrake.FunctionSpace(self.mesh, "DG", self.p - 1)  # the H(grad) space
                self.R_space = firedrake.FunctionSpace(self.mesh, 'R', 0)  # Lagrangian space for pressure to have integral 1
            else:
                raise TypeError("3D hexahedral meshes not available yet!")
            
        else:
            if (self.mesh.ufl_cell() == ufl.triangle):
                # For tetrahedral meshes
                self.U_space = firedrake.FunctionSpace(self.mesh, "RT", self.p)  # the H(div) space
                self.W_space = firedrake.FunctionSpace(self.mesh, "CG", self.p)  # the H(curl) space
                self.P_space = firedrake.FunctionSpace(self.mesh, "DG", self.p - 1)  # the L2 space
                self.R_Space = firedrake.FunctionSpace(self.mesh, 'R', 0)  # Lagrangian space for pressure to have integral 1
            else:
                # For quadrilateral meshes
                self.U_space = firedrake.FunctionSpace(self.mesh, "RTCF", self.p)  # the H(curl) space
                self.W_space = firedrake.FunctionSpace(self.mesh, "CG", self.p)  # the H(div) space
                self.P_space = firedrake.FunctionSpace(self.mesh, "DG", self.p - 1)  # the H(grad) space
                self.R_space = firedrake.FunctionSpace(self.mesh, 'R', 0)  # Lagrangian space for pressure to have integral 1

        # self.G_space = firedrake.FunctionSpace(self.mesh, "CG", self.p)  # the H(grad) space
        # self.C_space = firedrake.FunctionSpace(self.mesh, "N1curl", self.p)  # the H(curl) space
        # self.D_space = firedrake.FunctionSpace(self.mesh, "RT", self.p)  # the H(div) space
        # self.S_space = firedrake.FunctionSpace(self.mesh, "DG", self.p - 1)  # the L2 space
        # self.R_Space = firedrake.FunctionSpace(self.mesh, 'R', 0)  # Lagrangian space for pressure to have integral 1

        # Mixed function spaces
        self.mixed_space = self.U_space * self.W_space * self.P_space
        # self.mixed_space = self.D_space * self.C_space * self.S_space

        # Mixed null space to solve more efficiently for pressure
        self.p_nullspace = firedrake.VectorSpaceBasis(constant=True)
        self.mixed_nullspace = firedrake.MixedVectorSpaceBasis(self.mixed_space,
                                                               [self.mixed_space.sub(0), self.mixed_space.sub(1),
                                                                self.p_nullspace])
        # Trial and test functions
        self.epsilon2_trial, self.epsilon1_trial, self.epsilon3_trial = firedrake.TrialFunctions(self.mixed_space)
        self.epsilon2_test, self.epsilon1_test, self.epsilon3_test = firedrake.TestFunctions(self.mixed_space)

        # Solution variables
        # Mixed system functions with solutions at different time instants
        self.mixed_variables_0 = firedrake.Function(self.mixed_space)  # the mixed dual functions at initial step: velocity, vorticity, pressure
        self.mixed_variables_1 = firedrake.Function(self.mixed_space)  # the mixed dual functions at next step: velocity, vorticity, pressure

        # Individual functions with solutions at different time instants
        # Dual variables
        self.p_1 = self.mixed_variables_1.sub(2)  # pressure, 1 means integer step k + 1

        self.u_1_2 = self.mixed_variables_0.sub(0)  # velocity, 1_2 means half step k + 1/2
        self.u_3_2 = self.mixed_variables_1.sub(0)  # velocity, 3_2 means half step k + 3/2

        self.w_1_2 = self.mixed_variables_0.sub(1)  # vorticity, 1_2 means half step k + 1/2
        self.w_3_2 = self.mixed_variables_1.sub(1)  # vorticity, 3_2 means half step k + 3/2

    def __init_solver_parameters(self):
        # Solver parameters
        # self.solver_parameters = {'ksp_type': 'preonly',
        #                      'pc_type': 'lu',
        #                      # 'mat_type': 'aij',
        #                      'pc_factor_mat_solver_type': 'mumps',
        #                      # 'ksp_monitor_true_residual': None,
        #                     #  'ksp_initial_guess_nonzero': True,
        #                      'ksp_converged_reason': None}

        self.solver_parameters = {'ksp_type': 'gmres',
                             'pc_type': 'asm',
                             # 'mat_type': 'aij',
                             'pc_factor_mat_solver_type': 'mumps',
                             # 'ksp_monitor_true_residual': None,
                             'ksp_initial_guess_nonzero': True,
                             'ksp_converged_reason': None}

        # self.solver_parameters = {"pc_type": "fieldsplit",
        #                      "pc_fieldsplit_type": "multiplicative",
        #                      # first split contains first two fields, second
        #                      # contains the third
        #                      # 'mat_type': 'aij',
        #                      "pc_fieldsplit_0_fields": "0, 2",
        #                      "pc_fieldsplit_1_fields": "1",
        #                      # Multiplicative fieldsplit for first field
        #                      "fieldsplit_0_pc_type": "fieldsplit",
        #                      "fieldsplit_0_pc_fieldsplit_type": "schur",
        #                      #  "fieldsplit_0_pc_fieldsplit_schur_fact_type": "lower",
        #                      # LU on each field
        #                      "fieldsplit_0_fieldsplit_0_pc_type": "ilu",
        #                      "fieldsplit_0_fieldsplit_1_pc_type": "jacobi",
        #                      # ILU on the schur complement block
        #                      "fieldsplit_1_pc_type": "ilu",
        #                      'ksp_initial_guess_nonzero': True,
        #                      # 'ksp_monitor_true_residual': None,
        #                      'ksp_converged_reason': None}

        # self.solver_parameters = {"pc_type": "fieldsplit",
        #                      "pc_fieldsplit_type": "multiplicative",
        #                      # first split contains first two fields, second
        #                      # contains the third
        #                      # 'mat_type': 'aij',
        #                      'ksp_rtol': 1e-10,
        #                      "pc_fieldsplit_0_fields": "0, 2",
        #                      "pc_fieldsplit_1_fields": "1",
        #                      # Multiplicative fieldsplit for first field
        #                      "fieldsplit_0_pc_type": "fieldsplit",
        #                      "fieldsplit_0_pc_fieldsplit_type": "schur",
        #                      #  "fieldsplit_0_pc_fieldsplit_schur_fact_type": "lower",
        #                      # LU on each field
        #                      "fieldsplit_0_fieldsplit_0_pc_type": "ilu",
        #                      "fieldsplit_0_fieldsplit_1_pc_type": "jacobi",
        #                      # ILU on the schur complement block
        #                      "fieldsplit_1_pc_type": "ilu",
        #                      'ksp_initial_guess_nonzero': True,
        #                      # 'ksp_monitor_true_residual': None,
        #                      'ksp_converged_reason': None}


def cross(u, v):
    # Computes the cross product of two fields, they may be both vector fields, or one a vector field (2D) and the other a scalar field
    
    # Check which case we are: 
    #   3D vector field x 3D vector field
    #   2D vector field x scalar field
    # and compute the curl accordingly
    if ((u.ufl_shape == (3,)) and (v.ufl_shape == (3,))) or ((u.ufl_shape == (2,)) and (v.ufl_shape == (2,))):
        # Vector x Vector
        cross_product = firedrake.cross(u, v)

    elif ((u.ufl_shape == (2,)) and (v.ufl_shape == ())):
        # Vector (2D) x scalar (e_z)
        cross_product = firedrake.as_vector([u[1], -u[0]]) * v

    elif ((u.ufl_shape == ()) and (v.ufl_shape == (2,))):
        # scalar (e_z) x Vector (2D)
        cross_product = firedrake.as_vector([-u[1], u[0]]) * v

    else:
        raise TypeError("Curl product only allowed between two vectors or a 2D vector and a scalar field.")
    
    return cross_product