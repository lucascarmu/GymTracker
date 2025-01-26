import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Datos de ejemplo: puntos (x, y, z)
points = [
    (1, 2, 3),
    (4, 5, 6),
    (7, 8, 9),
    (2, 4, 6),
    (3, 6, 9)
]

# Separar los puntos en listas de coordenadas x, y, z
x_coords = [point[0] for point in points]
y_coords = [point[1] for point in points]
z_coords = [point[2] for point in points]

# Crear una figura y un objeto 3D
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Graficar los puntos en 3D
ax.scatter(x_coords, y_coords, z_coords, c='r', marker='o')  # Color rojo, marcador círculo

# Etiquetas de los ejes
ax.set_xlabel('Eje X')
ax.set_ylabel('Eje Y')
ax.set_zlabel('Eje Z')

# Título del gráfico
ax.set_title('Puntos en 3D')

# Mostrar el gráfico
plt.show()
