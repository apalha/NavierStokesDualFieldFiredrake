import firedrake
import math
import dual_field

# Export to file
primal_velocity_outfile = firedrake.File("output/primal_velocity.pvd")
primal_vorticity_outfile = firedrake.File("output/primal_vorticity.pvd")
dual_velocity_outfile = firedrake.File("output/dual_velocity.pvd")
dual_vorticity_outfile = firedrake.File("output/dual_vorticity.pvd")

# Physical domain and mesh
periodic = True
geometry_3d = False  # True: 3D simulation, False: 2D simulation
n_elements = [40, 40, 3]
bounds = [2.0, 2.0, 2.0]

# Flow characteristics
inv_Rf = 0.1  # the inverse of the fluid Reynolds number

# Temporal approximation
t = 0.0  # the start time instant
dt = 0.1  # the time step size
nt_steps = 4  # number of time steps

# Mesh
if periodic:
    if geometry_3d:
        mesh = firedrake.PeriodicBoxMesh(n_elements[0], n_elements[1], n_elements[2], bounds[0], bounds[1], bounds[2])
    else:
        mesh = firedrake.PeriodicRectangleMesh(n_elements[0], n_elements[1], bounds[0], bounds[1], quadrilateral=True)
else:
    if geometry_3d:
        mesh = firedrake.BoxMesh(n_elements[0], n_elements[1], n_elements[2], bounds[0], bounds[1], bounds[2])
    else:
        mesh = firedrake.RectangleMesh(n_elements[0], n_elements[1], bounds[0], bounds[1], quadrilateral=True)

# Analytical expressions
# Variables
if geometry_3d:
    x, y, z = firedrake.SpatialCoordinate(mesh)
else:
    x, y = firedrake.SpatialCoordinate(mesh)
time = firedrake.Constant(t)

# Expressions to check exact solution of NS equations
# velocityExpression = firedrake.as_vector(
#     [(2.0 - time) * firedrake.cos(2.0 * math.pi * z), (1.0 + time) * firedrake.sin(2.0 * math.pi * z),
#     (1.0 - time) * firedrake.sin(2.0 * math.pi * x)])
# velocity_dtExpression = firedrake.as_vector(
#     [-firedrake.cos(2.0 * math.pi * z), firedrake.sin(2.0 * math.pi * z), -firedrake.sin(2.0 * math.pi * x)])

if geometry_3d:
    velocityExpression = firedrake.as_vector(
        [-firedrake.sin(math.pi * x) * firedrake.cos(math.pi * y) * firedrake.exp(-2.0 * math.pi * math.pi * inv_Rf * time), firedrake.cos(math.pi * x) * firedrake.sin(math.pi * y) * firedrake.exp(-2.0 * math.pi * math.pi * inv_Rf * time),
        (x/x) - 1.0])
    velocity_dtExpression = firedrake.as_vector(
        [2.0 * math.pi * math.pi * inv_Rf * firedrake.sin(math.pi * x) * firedrake.cos(math.pi * y) * firedrake.exp(-2.0 * math.pi * math.pi * inv_Rf * time), -2.0 * math.pi * math.pi * inv_Rf * firedrake.cos(math.pi * x) * firedrake.sin(math.pi * y) * firedrake.exp(-2.0 * math.pi * math.pi * inv_Rf * time),
        (x/x) - 1.0])
else:
    velocityExpression = firedrake.as_vector(
        [-firedrake.sin(math.pi * x) * firedrake.cos(math.pi * y) * firedrake.exp(-2.0 * math.pi * math.pi * inv_Rf * time), firedrake.cos(math.pi * x) * firedrake.sin(math.pi * y) * firedrake.exp(-2.0 * math.pi * math.pi * inv_Rf * time)])
    velocity_dtExpression = firedrake.as_vector(
        [2.0 * math.pi * math.pi * inv_Rf * firedrake.sin(math.pi * x) * firedrake.cos(math.pi * y) * firedrake.exp(-2.0 * math.pi * math.pi * inv_Rf * time), -2.0 * math.pi * math.pi * inv_Rf * firedrake.cos(math.pi * x) * firedrake.sin(math.pi * y) * firedrake.exp(-2.0 * math.pi * math.pi * inv_Rf * time)])

pExpression = 0.25 * (firedrake.cos(2.0 * math.pi * x) + firedrake.cos(2.0 * math.pi * y)) * firedrake.exp(-2.0 * math.pi * math.pi * inv_Rf * time)

pTotalExpression = 0.5 * firedrake.inner(velocityExpression, velocityExpression) + pExpression

vorticityExpression = firedrake.curl(velocityExpression)

fExpression = velocity_dtExpression - dual_field.cross(velocityExpression, vorticityExpression) + firedrake.grad(pTotalExpression) + inv_Rf * firedrake.curl(vorticityExpression)

# Initialize Navier-Stokes solvers

# First the objects
ns_primal = dual_field.NSPrimal(mesh, dt=dt, inv_Rf=inv_Rf)
ns_dual = dual_field.NSDual(mesh, dt=dt, inv_Rf=inv_Rf)

# Initialize the weak forms by linking them together
print('Initializing primal weak forms...')
ns_primal.init_weak_forms(ns_dual.w_1_2, f_expression=fExpression)
print('Initializing dual weak forms...')
ns_dual.init_weak_forms(ns_primal.w_1, f_expression=fExpression)

# Set the initial conditions
# Primal variables are set at the time step t = 0
print('Setting primal variables initial conditions...')
t = 0.0
time.assign(t)
ns_primal.set_initial_conditions(velocityExpression, vorticityExpression)

# Dual variables are set at time step t = 1/2
print('Setting dual variables initial conditions...')
t = 0.5*dt
time.assign(t)
ns_dual.set_initial_conditions(velocityExpression, vorticityExpression)

t = 0.0  # reset the time step

# Save solutions to file 
# Primal system 
primal_velocity_outfile.write(ns_primal.u_0, time=t)  # note that this is the integer time instant 
primal_vorticity_outfile.write(ns_primal.w_0, time=t)
# Dual system 
dual_velocity_outfile.write(ns_dual.u_1_2, time=t + 0.5*dt)  # note that this is the fractional time instant 
dual_vorticity_outfile.write(ns_dual.w_1_2, time=t + 0.5*dt)

# Compute errors
# Primal system
time.assign(t)  # update the time variable note that this is the integer time instant 
error_u_primal = ns_primal.error_velocity(velocityExpression)
error_w_primal = ns_primal.error_vorticity(vorticityExpression)
print("   Primal:")
print(f"      error u: {error_u_primal}")
print(f"      error u: {error_w_primal}")
# Dual system
time.assign(t + 0.5*dt)  # update the time variable note that this is the fractional time instant 
error_u_dual = ns_dual.error_velocity(velocityExpression)
error_w_dual = ns_dual.error_vorticity(vorticityExpression)
print("   Dual:")
print(f"      error u: {error_u_dual}")
print(f"      error u: {error_w_dual}")

# Now time step
for t_step_idx in range(1, nt_steps + 1):
    # Update time
    t += 0.5 * dt  # update time step
    time.assign(t)  # update the time variable
    print('Time step: %f' % (t + 0.5 * dt))

    # Leap frog advance the primal system
    ns_primal.time_step()

    # Update time
    t += 0.5 * dt
    time.assign(t)  # update the time variable
    print('Time step: %f' % (t + 0.5 * dt))

    # Leap frog advance the dual system
    ns_dual.time_step()

    # Save solutions to file 
    # Primal system 
    primal_velocity_outfile.write(ns_primal.u_0, time=t)  # note that this is the integer time instant 
    primal_vorticity_outfile.write(ns_primal.w_0, time=t)
    # Dual system 
    dual_velocity_outfile.write(ns_dual.u_1_2, time=t + 0.5*dt)  # note that this is the fractional time instant 
    dual_vorticity_outfile.write(ns_dual.w_1_2, time=t + 0.5*dt)

    # Compute errors
    # Primal system
    time.assign(t)  # update the time variable note that this is the integer time instant 
    error_u_primal = ns_primal.error_velocity(velocityExpression)
    error_w_primal = ns_primal.error_vorticity(vorticityExpression)
    print("   Primal:")
    print(f"      error u: {error_u_primal}")
    print(f"      error u: {error_w_primal}")
    # Dual system
    time.assign(t + 0.5*dt)  # update the time variable note that this is the fractional time instant 
    error_u_dual = ns_dual.error_velocity(velocityExpression)
    error_w_dual = ns_dual.error_vorticity(vorticityExpression)
    print("   Dual:")
    print(f"      error u: {error_u_dual}")
    print(f"      error u: {error_w_dual}")

print('Finished...')
