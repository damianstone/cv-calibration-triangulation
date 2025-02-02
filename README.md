# Mock Data Generator

## Repository Structure
```markdown
mock-data-generator/
├── data/                     # Datos generados y pruebas
│   ├── input/                # Datos iniciales (x, y)
│   └── output/               # Mock data generada (x, y, z)
├── src/                      # Código fuente
│   ├── generate_mock.py      # Genera datos mock (x, y, z)
│   ├── filters.py            # Filtra candidatos válidos
│   └── trajectories.py       # Genera trayectorias probables
├── notebooks/                # Prototipos y visualización
│   └── mock_data_visualizer.ipynb
├── tests/                    # Pruebas unitarias
│   ├── test_generate_mock.py
│   ├── test_filters.py
│   └── test_trajectories.py
├── utils/                    # Funciones auxiliares
│   └── plotting.py           # Graficar mock data y trayectorias
├── README.md                 # Descripción del proyecto
└── .gitignore                # Archivos a ignorar por Git
```
