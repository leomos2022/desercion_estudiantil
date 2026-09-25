import matplotlib.pyplot as plt
import networkx as nx

# Definir nodos
actividades = [
    "Verificación de Entregables",
    "Aceptación Formal del Cliente",
    "Liquidación Financiera",
    "Lecciones Aprendidas",
    "Liberación de Recursos",
    "Archivo de Documentación",
    "Cierre de Contratos"
]
documentos = [
    "Acta de Cierre",
    "Lista de Verificación",
    "Informe de Lecciones"
]

# Crear grafo dirigido
G = nx.DiGraph()

# Agregar nodos
for act in actividades:
    G.add_node(act, group='Actividad')
for doc in documentos:
    G.add_node(doc, group='Documento')

# Agregar relaciones
G.add_edge("Verificación de Entregables", "Acta de Cierre")
G.add_edge("Aceptación Formal del Cliente", "Acta de Cierre")
G.add_edge("Liquidación Financiera", "Acta de Cierre")
G.add_edge("Liberación de Recursos", "Acta de Cierre")
G.add_edge("Cierre de Contratos", "Acta de Cierre")
G.add_edge("Lecciones Aprendidas", "Lista de Verificación")
G.add_edge("Lecciones Aprendidas", "Informe de Lecciones")
G.add_edge("Archivo de Documentación", "Lista de Verificación")

# Posiciones para los nodos
pos = {
    "Verificación de Entregables": (-2, 2),
    "Aceptación Formal del Cliente": (-1, 3),
    "Liquidación Financiera": (0, 2.5),
    "Lecciones Aprendidas": (1, 3),
    "Liberación de Recursos": (2, 2),
    "Archivo de Documentación": (1, 1),
    "Cierre de Contratos": (0, 1),
    "Acta de Cierre": (0, 0),
    "Lista de Verificación": (2, 0),
    "Informe de Lecciones": (3, 1.5)
}

# Colores
color_map = []
for node in G:
    if G.nodes[node]['group'] == 'Actividad':
        color_map.append('#1976D2')  # Azul
    else:
        color_map.append('#43A047')  # Verde

plt.figure(figsize=(12, 7))
nx.draw(G, pos, with_labels=True, node_color=color_map, node_size=3200, font_size=10, font_weight='bold', arrowsize=20, arrowstyle='-|>')
plt.title("Diagrama de Actividades y Documentos Clave para el Cierre de Proyecto", fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig("grafico_cierre_proyecto.png", dpi=300)
plt.show()
