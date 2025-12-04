# --- PROTOCOLO GEOMETROGÉNESIS v14 ---
# Objetivo: Calibrar el universo simulado para encontrar ns = 0.9649.
# Este script unificado permite ejecutar diferentes fases del protocolo
# de calibración, ajustando parámetros cuánticos para acercarse al valor objetivo.

# --- Paso 1: Importar las herramientas necesarias ---
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
from qiskit.visualization import plot_histogram
import matplotlib.pyplot as plt
import numpy as np

# --- Configuración del Protocolo ---
FASE_A_EJECUTAR = 5 # <--- CAMBIA AQUÍ LA FASE QUE QUIERES EJECUTAR (2, 3, o 5)

# --- Parámetros Globales ---
TARGET_NS = 0.9649
SHOTS = 8192

# --- Parámetros por Fase ---
# Estos son los parámetros que se explorarán en cada fase.
parametros_fase = {
    2: {
        "nombre": "FASE 2: BARRIDO DE INTERACCIÓN CUÁNTICA (J_COUPLING)",
        "variable_nombre": "J_Coupling",
        "variable_valores": [-0.01, -0.03, -0.05, -0.07],
        "parametros_fijos": {"COUPLING_STRENGTH": 0.9},
        "success_threshold": 0.01
    },
    3: {
        "nombre": "FASE 3: BARRIDO DE CAMPO TRANSVERSO (H_TRANSVERSE)",
        "variable_nombre": "H_Transverse",
        "variable_valores": [-0.1, -1.0, -2.0],
        "parametros_fijos": {"COUPLING_STRENGTH": 0.9, "J_COUPLING": -0.01},
        "success_threshold": 0.01
    },
    5: {
        "nombre": "FASE 5: EL LÍMITE CLÁSICO",
        "variable_nombre": "H_Transverse",
        "variable_valores": [-0.01, -0.03, -0.05, -0.07, -0.09],
        "parametros_fijos": {"COUPLING_STRENGTH": 0.9, "J_COUPLING": -0.01},
        "success_threshold": 0.005
    }
}

def ejecutar_fase(num_fase):
    """
    Ejecuta una fase completa del protocolo Geometrogénesis.
    """
    config = parametros_fase.get(num_fase)
    if not config:
        print(f"Error: La fase {num_fase} no está definida.")
        return

    # --- Preparación del Laboratorio ---
    print(f"\n--- INICIANDO PROTOCOLO GEOMETROGÉNESIS: {config['nombre']} ---\n")
    for nombre, valor in config["parametros_fijos"].items():
        print(f"Parámetro base fijado: {nombre} = {valor}")
    print("-" * 40)

    variable_valores = config["variable_valores"]
    num_simulaciones = len(variable_valores)
    resultados_ns = {}

    fig, axes = plt.subplots(num_simulaciones, 1, figsize=(10, 5 * num_simulaciones), squeeze=False)
    fig.suptitle(f'GEOMETROGÉNESIS {config["nombre"]}\n(Objetivo n_s={TARGET_NS}, |error| < {config["success_threshold"]})', fontsize=16)

    simulador = AerSimulator()

    # --- Inicio de la Simulación por Pasos ---
    for i, valor_variable in enumerate(variable_valores):
        circuito = QuantumCircuit(1, 1)
        
        # Parámetros para las compuertas
        coupling = config["parametros_fijos"].get("COUPLING_STRENGTH", 0)
        j_coupling = config["parametros_fijos"].get("J_COUPLING", 0)
        h_transverse = 0

        if config["variable_nombre"] == "J_Coupling":
            j_coupling = valor_variable
        elif config["variable_nombre"] == "H_Transverse":
            h_transverse = valor_variable

        # 1. Compuerta de Acoplamiento (RY)
        if coupling != 0:
            circuito.ry(coupling * np.pi, 0)

        # 2. Compuerta de Interacción (RZ)
        if j_coupling != 0:
            circuito.rz(j_coupling * np.pi, 0)

        # 3. Compuerta de Campo Transverso (RX)
        if h_transverse != 0:
            circuito.rx(h_transverse * np.pi, 0)

        # 4. Medición
        circuito.measure(0, 0)

        # 5. Simulación
        circuito_compilado = transpile(circuito, simulador)
        trabajo = simulador.run(circuito_compilado, shots=SHOTS)
        conteos = trabajo.result().get_counts(circuito_compilado)

        # 6. Cálculo de n_s
        prob_1 = conteos.get('1', 0) / SHOTS
        resultados_ns[valor_variable] = prob_1
        diferencia = abs(prob_1 - TARGET_NS)
        
        print(f"{config['variable_nombre']}={valor_variable:.3f} -> ns resultante={prob_1:.4f} (Diferencia: {diferencia:.5f})")

        # 7. Visualización
        ax = axes[i, 0]
        plot_histogram(conteos, ax=ax, title=f"{config['variable_nombre']} = {valor_variable:.3f} | $n_s$ = {prob_1:.4f}")
        ax.title.set_size(12)

    # --- Análisis de Resultados ---
    print("\n--- ANÁLISIS DE RESULTADOS ---")
    valor_ganador = min(resultados_ns, key=lambda k: abs(resultados_ns[k] - TARGET_NS))
    ns_ganador = resultados_ns[valor_ganador]
    diferencia_final = abs(ns_ganador - TARGET_NS)

    print(f"Mejor valor encontrado: {config['variable_nombre']} = {valor_ganador}")
    print(f"n_s obtenido: {ns_ganador:.4f}")
    print(f"Diferencia mínima con el objetivo: {diferencia_final:.5f}")

    if diferencia_final < config["success_threshold"]:
        print(f"\nESTADO: ¡ÉXITO! Diferencia < {config['success_threshold']}. PROTOCOLO COMPLETADO. ¡UNIVERSO CALIBRADO!")
    else:
        print(f"\nESTADO: Fase {num_fase} finalizada. No se alcanzó el objetivo de precisión.")

    # --- Finalización y Muestra de Gráficos ---
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

if __name__ == "__main__":
    ejecutar_fase(FASE_A_EJECUTAR)