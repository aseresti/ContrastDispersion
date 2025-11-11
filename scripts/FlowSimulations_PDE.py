import fenics as fe
import numpy as np
import matplotlib.pyplot as plt

if __name__ == "__main__":
    n_elements = 100
    L = 1.0
    Tfinal = 30.0
    target_CFL = 2.0
    mesh = fe.UnitIntervalMesh(n_elements)

    # Define velocity and diffusion coefficient
    velocity = fe.Constant((2.0,))
    diffusion = fe.Constant(0.04)

    # Define function space
    lagrange_polynomial_space_first_order = fe.FunctionSpace(
        mesh, "Lagrange", 1
    )

    # Define time-dependent boundary condition at x=0 using a Gaussian pulse
    A = 800.0
    t0 = 15.0
    sigma = 2.5

    u_D = fe.Constant(0.0)

    def inlet_value(t):
        return A*np.exp(-0.5*((t - t0)/sigma)**2)
    
    def tac_gamma(t, K=1.0, t0=3.0, alpha=3.0, beta=1.5):
        if t <= t0:
            return 0.0
        x = (t - t0) / beta
        return K * (x**alpha) * np.exp(-x)

    # Define boundary condition function to return whether we are on the boundary
    def boundary_boolean_function(x, on_boundary):
        return on_boundary and fe.near(x[0], 0.0)
    
    # The homogeneous Dirichlet boundary condition
    boundary_condition = fe.DirichletBC(
        lagrange_polynomial_space_first_order,
        u_D,
        boundary_boolean_function,
    )

    # Define initial condition
    u_old = fe.Function(lagrange_polynomial_space_first_order)
    u_old.vector()[:] = 0.0

    # Define time stepping of implicit Euler method (=dt)
    h = L / n_elements
    dt = 1e-2#target_CFL * h / velocity.values()[0]
    n_steps = int(np.ceil(Tfinal / dt))

    # The force on the right-hand side
    f = fe.Constant(0.0)

    # Create the Finite Element variational problem
    u = fe.TrialFunction(lagrange_polynomial_space_first_order)
    v = fe.TestFunction(lagrange_polynomial_space_first_order)


    # Weak form of the Advection-Diffusion equation
    weak_form_residuum = (
        u * v * fe.dx
        +
        dt * fe.dot(velocity, fe.grad(u)) * v * fe.dx
        +
        dt * diffusion * fe.dot(fe.grad(u), fe.grad(v)) * fe.dx
        -
        (
            u_old * v * fe.dx
            +
            dt * f * v * fe.dx
        )
    )

    # Convert to linear system
    weak_form_lhs = fe.lhs(weak_form_residuum)
    weak_form_rhs = fe.rhs(weak_form_residuum)

    # Prepare solution function
    u_solution = fe.Function(
        lagrange_polynomial_space_first_order
    )

    # Time-stepping loop
    t_current = 0.0

    # Store final solution for plotting
    u_final = np.zeros((n_elements + 1, n_steps + 1))
    u_final[:, 0] = u_old.vector().get_local()

    for i in range(n_steps):
        t_current += dt
        u_D.assign(tac_gamma(t_current))

        # Assemble system, BC applied here
        fe.solve(
            weak_form_lhs == weak_form_rhs,
            u_solution,
            boundary_condition,
        )

        # Update for next time step
        u_old.assign(u_solution)

        # Store solution
        u_final[:, i + 1] = u_solution.vector().get_local()

    # Plot results as an image
    x = np.linspace(0, 1, n_elements + 1)
    T = np.linspace(0, n_steps * dt, n_steps + 1)
    X, TT = np.meshgrid(x, T, indexing='ij')

    plt.imshow(u_final, extent=[0,1,0,n_steps*dt], origin='lower', aspect='auto')
    plt.colorbar(label="u(x,t)")
    plt.title("1D Advection-Diffusion Equation Solution over Time")
    plt.xlabel("x")
    plt.ylabel("t")
    plt.show()
        