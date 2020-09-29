
# Equidistant

A program that finds equidistant points. Given the number of points and the number of dimensions, the program tries to find a configuration of points that equalizes all distances between them. This constraint is specified as a loss function and optimized using PyTorch. I like the engineering simplicity.

    model = gc.GeometricConstraints(points=3, dimensions=2, constraints=[
        lambda points, distances: distances.var()
    ])

    plt.scatter(*model.points().T)

![](images/3p2d_001.svg)

According to the internet, full equidistance can only be achieved if num_points <= num_dimensions + 1. The more interesting results are when we go past this constraint.

    model = gc.GeometricConstraints(points=5, dimensions=2, iterations=10000, constraints=[
        lambda points, distances: distances.var() + 1e-6 * (1 / distances**2).sum()
    ])

    points = model.points()
    points -= points.mean(axis=0)
    plt.scatter(*points.T);

![](images/5p2d_001.svg)

Here we've mean-centered the points and added an anti-singularity constraint `1e-6 * (1 / distances**2).sum()` that keeps the points from collapsing into each other, which is something that begins happening with larger numbers of particles. This is partly because we're constraining the distances by their variance and not a specific number.

We can constrain nice symmetries directly:

    model = gc.GeometricConstraints(points=5, dimensions=2, iterations=10000, constraints=[
        lambda points, distances: distances.var() + 1e-6 * (1 / distances**2).sum(),
        lambda points, distances: points[:2, 1].var()
    ])

![](images/5p2d_002_symmetry.svg)

The second constraint wants two points to have the same y value, which gives the figure a left-right symmetry. Sometimes the figures fall into different local minima:

![](images/5p2d_002_symmetry_2.svg)

Points can be explicitly constrained:

    model = gc.GeometricConstraints(points=11, dimensions=2, iterations=10000, constraints=[
        lambda points, distances: distances.var() + 1e-6 * (1 / distances**2).sum(),
        lambda points, distances: ((points[:2] - torch.tensor([[0, -1], [0, 1]]))**2).sum()
    ])

    points = model.points()
    plt.scatter(*points[:2].T)
    plt.scatter(*points[2:].T);

![](images/11p2d_001.svg)

This could always be done by hard coding the points, but specifying them as constraints works well, and the weight on the constraint can always be increased to make it resemble a hard coded value.

It's fun to pin different locations to see what the system does:

    model = gc.GeometricConstraints(points=11, dimensions=2, iterations=10000, constraints=[
        lambda points, distances: distances.var() + 1e-6 * (1 / distances**2).sum(),
        lambda points, distances: ((points[:4] - torch.tensor([[-1, 0], [1, 0], [0, -1], [0, -2]]))**2).sum()
    ])

    points = model.points()
    plt.scatter(*points[:4].T, color='orange');
    plt.scatter(*points[4:].T);

![](images/11p2d_002_asymmetry.svg)

    triangle = torch.tensor([[np.sqrt(3), -1], [-np.sqrt(3), -1], [0, 8]])
    model = gc.GeometricConstraints(points=41, dimensions=2, iterations=40000, constraints=[
        lambda points, distances: distances.var() + 1e-6 * (1 / distances**2).sum(),
        lambda points, distances: ((points[:3] - triangle)**2).sum()
    ])

    points = model.points()
    plt.scatter(*points[:3].T, s=20, color='orange');
    plt.scatter(*points[3:].T, s=20);

![](images/41p2d_001.svg)

Since the system isn't able to fully satisfy the equidistance constraint with a larger number of particles, we can look at some of the tradeoffs it makes by plotting the pairwise distances alongside the configurations:

![](images/3p_to_10p_distances_histogram.svg)