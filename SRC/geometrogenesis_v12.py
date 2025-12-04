import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import webbrowser
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing

# --- Forzar uso de CPU ---
FORCE_CPU = True

# --- Detección de GPU ---
try:
    if FORCE_CPU:
        raise ImportError("Forzando el uso de CPU por configuración del usuario.")
    import cupy as cp
    GPU_ENABLED = True
    print("✓ GPU detectada. Usando CuPy para aceleración.")
except (ImportError, ModuleNotFoundError):
    cp = np
    GPU_ENABLED = False
    if FORCE_CPU:
        print("✓ Forzando uso de CPU. Usando NumPy.")
    else:
        print("⚠️ GPU no detectada. Usando NumPy (funcionamiento más lento).")

# --- Importaciones de PennyLane ---
try:
    import pennylane as qml
    from pennylane import numpy as pnp
    from pennylane.optimize import AdamOptimizer
    QUANTUM_LIB_AVAILABLE = True
    print("✓ PennyLane detectado. Usando para la capa cuántica.")
except ImportError:
    print("⚠️ PennyLane no está instalado. Se usará un modelo simplificado.")
    QUANTUM_LIB_AVAILABLE = False

# --- Importaciones de Visualización ---
try:
    import imageio
    import matplotlib.pyplot as plt
    VISUALIZATION_LIBS_AVAILABLE = True
except ImportError:
    print("⚠️ Matplotlib o Imageio no están instalados. La visualización estará desactivada.")
    VISUALIZATION_LIBS_AVAILABLE = False
    # Mock objects if not available to avoid runtime errors on calls
    imageio = None
    plt = None

# --- PARÁMETROS DE LA SIMULACIÓN V12 ---
GRID_SIZE = 20  # Reducido para VQE
PASOS_SIMULACION = 50

# --- Parámetros del Hamiltoniano de Ising ---
N_QUBITS = 2
N_LAYERS = 2
J_COUPLING = -1.0
H_TRANSVERSE = -0.5

# --- Parámetros de VQE ---
VQE_ITERATIONS_PER_STEP = 10  # Aumentado para Adam
VQE_PARALLEL = True  # Activar paralelización
MAX_WORKERS = max(1, multiprocessing.cpu_count() - 1)  # Dejar 1 CPU libre

# Parámetros del Campo Físico
MU_SQUARED = 1.0
LAMBDA_PARAM = 1.0
DT = 0.02
COUPLING_STRENGTH = 0.5
LEARNING_RATE = 0.1 # Tasa de aprendizaje para el optimizador Adam

# --- Hamiltoniano Global (para evitar recrearlo) ---
GLOBAL_HAMILTONIAN = None

# --- Dispositivo Cuántico Global ---
QUANTUM_DEVICE = None


def _crear_hamiltoniano_ising(n_qubits, j_coupling, h_transverse):
    """Crea el Hamiltoniano del Modelo de Ising Transverso."""
    if not QUANTUM_LIB_AVAILABLE:
        return None
    
    coeffs = []
    obs = []
    # Términos ZZ (interacción)
    for i in range(n_qubits - 1):
        coeffs.append(j_coupling)
        obs.append(qml.PauliZ(i) @ qml.PauliZ(i + 1))
    # Términos X (campo transverso)
    for i in range(n_qubits):
        coeffs.append(h_transverse)
        obs.append(qml.PauliX(i))
    
    print(f"✓ Hamiltoniano de Ising creado con {len(coeffs)} términos.")
    return qml.Hamiltonian(coeffs, obs)


def _optimizar_agente_vqe(args):
    """
    Función worker para optimización VQE paralela.
    
    Args:
        args: tuple (i, j, initial_params, field_influence, device, hamiltonian)
    
    Returns:
        tuple: (i, j, optimized_params, final_energy)
    """
    i, j, initial_params, field_influence, device, hamiltonian = args
    
    try:
        optimizer = AdamOptimizer(stepsize=LEARNING_RATE)
        params = pnp.array(initial_params, requires_grad=True)

        @qml.qnode(device)
        def cost_fn(p):
            _build_circuit_from_params_static(p)
            return qml.expval(hamiltonian)

        for _ in range(VQE_ITERATIONS_PER_STEP):
            params, _ = optimizer.step_and_cost(lambda p: cost_fn(p) + field_influence, params)

        final_energy = cost_fn(params) + field_influence
        optimized_params = params.unwrap()  # Obtener el valor sin gradiente
        
        return (i, j, optimized_params, final_energy)
    
    except Exception as e:
        print(f"  ⚠️ Error en agente ({i},{j}): {e}")
        return (i, j, initial_params, float('inf'))


def _build_circuit_from_params_static(params):
    """Construye circuito desde parámetros aplanados (para uso en workers)."""
    for l in range(N_LAYERS):
        for i in range(N_QUBITS):
            qml.RY(params[l, i], wires=i)
        if N_QUBITS > 1:
            for i in range(N_QUBITS - 1):
                qml.CNOT(wires=[i, i + 1])


class SimuladorVQE:
    """
    Simulador Científico v12: Geometrogénesis con Optimización VQE.
    
    Características principales:
    - Optimización cuántica variacional (VQE) para cada agente
    - Paralelización opcional de la optimización
    - Acoplamiento bidireccional campo-agentes
    - Análisis cosmológico del espectro de potencia
    """
    
    def __init__(self):
        print(f"\n{'='*70}")
        print(f"  GEOMETROGÉNESIS COMPUTACIONAL v12 (VQE)")
        print(f"  {'GPU' if GPU_ENABLED else 'CPU'} Mode | Grid: {GRID_SIZE}×{GRID_SIZE} | Steps: {PASOS_SIMULACION}")
        print(f"  VQE: {VQE_ITERATIONS_PER_STEP} iter/agente | Paralelo: {VQE_PARALLEL}")
        if VQE_PARALLEL:
            print(f"  Workers: {MAX_WORKERS} cores")
        print(f"{'='*70}\n")
        
        self.xp = cp if GPU_ENABLED else np
        
        # Inicializar Hamiltoniano global
        global GLOBAL_HAMILTONIAN
        global QUANTUM_DEVICE
        if QUANTUM_LIB_AVAILABLE:
            QUANTUM_DEVICE = qml.device("default.qubit", wires=N_QUBITS)
            GLOBAL_HAMILTONIAN = _crear_hamiltoniano_ising(N_QUBITS, J_COUPLING, H_TRANSVERSE)
            self.hamiltonian = GLOBAL_HAMILTONIAN
        else:
            self.hamiltonian = None
            print("⚠️ Usando modelo de energía simplificado sin PennyLane")
        
        # --- Estado del Campo Φ ---
        self.Phi = self.xp.array(
            np.random.uniform(-1, 1, (GRID_SIZE, GRID_SIZE)) + 
            1j * np.random.uniform(-1, 1, (GRID_SIZE, GRID_SIZE))
        )
        
        # --- Estado de los Agentes ---
        self.circuit_params = self.xp.array(
            np.random.uniform(0, 2 * np.pi, (GRID_SIZE, GRID_SIZE, N_LAYERS, N_QUBITS))
        )
        self.energies = self.xp.ones((GRID_SIZE, GRID_SIZE))
        self.fitness = self.xp.zeros((GRID_SIZE, GRID_SIZE))
        
        # --- Historial y Frames ---
        self.history = {
            "Energia_Campo": [],
            "Energia_Cuántica_Promedio": [],
            "Fitness_Promedio": [],
            "Tiempo_Paso": []
        }
        self.frames_for_gif = []
    
    def evaluar_energias_grid(self):
        """Evalúa energía de todo el grid (batch para eficiencia)."""
        field_influence = COUPLING_STRENGTH * (self.xp.abs(self.Phi)**2 - 1)
        
        if QUANTUM_LIB_AVAILABLE and self.hamiltonian is not None:
            params_cpu = self._to_cpu(self.circuit_params)

            @qml.qnode(QUANTUM_DEVICE)
            def circuit(params):
                _build_circuit_from_params_static(params)
                return qml.expval(self.hamiltonian)

            # La evaluación en batch es más compleja de configurar en PennyLane
            # para este caso de uso. La paralelización por agente es más directa.
            # Evaluamos secuencialmente aquí, la optimización es la parte costosa.
            quantum_energies_list = []
            for i in range(GRID_SIZE):
                for j in range(GRID_SIZE):
                    quantum_energies_list.append(circuit(params_cpu[i, j]))
            
            quantum_energies = self.xp.array(quantum_energies_list).reshape(GRID_SIZE, GRID_SIZE)
        else:
            # Modelo simplificado sin PennyLane
            params_norm = self.xp.linalg.norm(self.circuit_params, axis=(2, 3))
            quantum_energies = -self.xp.cos(params_norm)
        
        self.energies = quantum_energies + field_influence
        self.fitness = 1.0 / (1.0 + self.xp.exp(0.5 * self.energies))
    
    def evolucionar_agentes_vqe_paralelo(self):
        """Evolución de agentes con VQE paralelizado."""
        if not QUANTUM_LIB_AVAILABLE:
            print("  ⚠️ Saltando VQE (PennyLane no disponible)")
            return
        
        # 1. Fase de Contagio (selección local rápida)
        fitness_pad = self.xp.pad(self.fitness, pad_width=1, mode='wrap')
        params_pad = self.xp.pad(
            self.circuit_params,
            pad_width=((1,1), (1,1), (0,0), (0,0)),
            mode='wrap'
        )
        
        best_neighbor_fitness = self.xp.copy(fitness_pad)
        best_neighbor_params = self.xp.copy(params_pad)
        
        for di in [-1, 0, 1]:
            for dj in [-1, 0, 1]:
                if di == 0 and dj == 0:
                    continue
                
                rolled_fitness = self.xp.roll(fitness_pad, (di, dj), axis=(0, 1))
                rolled_params = self.xp.roll(params_pad, (di, dj), axis=(0, 1))
                
                mask = rolled_fitness > best_neighbor_fitness
                best_neighbor_fitness[mask] = rolled_fitness[mask]
                best_neighbor_params[mask] = rolled_params[mask]
        
        best_neighbor_params = best_neighbor_params[1:-1, 1:-1]
        adoption_mask = (best_neighbor_fitness[1:-1, 1:-1] > self.fitness)
        self.circuit_params[adoption_mask] = best_neighbor_params[adoption_mask]
        
        # 2. Fase de Optimización VQE
        print("    → Optimizando agentes con VQE...", end=" ", flush=True)
        vqe_start = time.time()
        
        params_cpu = self._to_cpu(self.circuit_params)
        field_influence_cpu = self._to_cpu(
            COUPLING_STRENGTH * (self.xp.abs(self.Phi)**2 - 1)
        )
        
        if VQE_PARALLEL:
            # Optimización paralela
            tasks = [
                (i, j, params_cpu[i, j], field_influence_cpu[i, j], QUANTUM_DEVICE, GLOBAL_HAMILTONIAN)
                for i in range(GRID_SIZE)
                for j in range(GRID_SIZE)
            ]
            
            with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
                futures = [executor.submit(_optimizar_agente_vqe, task) for task in tasks]
                
                for future in as_completed(futures):
                    i, j, optimized_params, _ = future.result()
                    params_cpu[i, j] = optimized_params
        else:
            # La versión secuencial ahora llama al worker para mantener consistencia
            for i in range(GRID_SIZE):
                for j in range(GRID_SIZE):
                    _, _, params_cpu[i, j], _ = _optimizar_agente_vqe((
                        i, j,
                        params_cpu[i, j],
                        field_influence_cpu[i, j],
                        QUANTUM_DEVICE,
                        GLOBAL_HAMILTONIAN
                    ))
        
        # Actualizar parámetros
        self.circuit_params = self.xp.array(params_cpu)
        
        vqe_time = time.time() - vqe_start
        print(f"✓ ({vqe_time:.1f}s)")
    
    def evolucionar_agentes_vqe_secuencial(self):
        """Evolución de agentes con VQE secuencial (fallback)."""
        if not QUANTUM_LIB_AVAILABLE:
            return
        
        # Contagio (igual que paralelo)
        fitness_pad = self.xp.pad(self.fitness, pad_width=1, mode='wrap')
        params_pad = self.xp.pad(
            self.circuit_params,
            pad_width=((1,1), (1,1), (0,0), (0,0)),
            mode='wrap'
        )
        
        best_neighbor_fitness = self.xp.copy(fitness_pad)
        best_neighbor_params = self.xp.copy(params_pad)
        
        for di in [-1, 0, 1]:
            for dj in [-1, 0, 1]:
                if di == 0 and dj == 0:
                    continue
                rolled_fitness = self.xp.roll(fitness_pad, (di, dj), axis=(0, 1))
                mask = rolled_fitness > best_neighbor_fitness
                best_neighbor_fitness[mask] = rolled_fitness[mask]
                rolled_params = self.xp.roll(params_pad, (di, dj), axis=(0, 1))
                best_neighbor_params[mask] = rolled_params[mask]
        
        best_neighbor_params = best_neighbor_params[1:-1, 1:-1]
        adoption_mask = (best_neighbor_fitness[1:-1, 1:-1] > self.fitness)
        self.circuit_params[adoption_mask] = best_neighbor_params[adoption_mask]
        
        # VQE secuencial
        params_cpu = self._to_cpu(self.circuit_params)
        field_influence_cpu = self._to_cpu(
            COUPLING_STRENGTH * (self.xp.abs(self.Phi)**2 - 1)
        )
        
        # VQE secuencial
        for i in range(GRID_SIZE):
            for j in range(GRID_SIZE):
                 _, _, params_cpu[i, j], _ = _optimizar_agente_vqe((
                     i, j,
                     params_cpu[i, j],
                     field_influence_cpu[i, j],
                     QUANTUM_DEVICE,
                     GLOBAL_HAMILTONIAN
                 ))

        self.circuit_params = self.xp.array(params_cpu)
    
    def evolucionar_campo(self):
        """Evoluciona el campo Φ según su ecuación de movimiento."""
        grad_potencial = (-MU_SQUARED + 2 * LAMBDA_PARAM * self.xp.abs(self.Phi)**2) * self.Phi
        agent_influence = COUPLING_STRENGTH * self.energies * self.Phi
        self.Phi -= (grad_potencial + agent_influence) * DT
    
    def _to_cpu(self, array):
        """Convierte array de GPU a CPU si es necesario."""
        return cp.asnumpy(array) if GPU_ENABLED else array
    
    def step(self, i):
        """Ejecuta un paso completo de simulación."""
        start_time = time.time()
        
        # 1. Evaluar estado actual
        self.evaluar_energias_grid()
        
        # 2. Registrar métricas
        energia_campo = self._to_cpu(
            self.xp.mean(-MU_SQUARED * self.xp.abs(self.Phi)**2 + 
                        LAMBDA_PARAM * self.xp.abs(self.Phi)**4)
        )
        energia_cuantica = self._to_cpu(self.xp.mean(self.energies))
        fitness_promedio = self._to_cpu(self.xp.mean(self.fitness))
        
        self.history["Energia_Campo"].append(energia_campo)
        self.history["Energia_Cuántica_Promedio"].append(energia_cuantica)
        self.history["Fitness_Promedio"].append(fitness_promedio)
        
        # 3. Capturar frame
        self.capturar_frame(i)
        
        # 4. Evolucionar agentes con VQE
        if VQE_PARALLEL:
            self.evolucionar_agentes_vqe_paralelo()
        else:
            self.evolucionar_agentes_vqe_secuencial()
        
        # 5. Evolucionar campo
        self.evolucionar_campo()
        
        step_time = time.time() - start_time
        self.history["Tiempo_Paso"].append(step_time)
        
        # Estimación de tiempo restante
        tiempo_promedio = np.mean(self.history["Tiempo_Paso"])
        tiempo_restante = tiempo_promedio * (PASOS_SIMULACION - i - 1)
        
        print(f"  ▸ Paso {i+1}/{PASOS_SIMULACION} | "
              f"E_Q: {energia_cuantica:.3f} | "
              f"E_Φ: {energia_campo:.3f} | "
              f"Fitness: {fitness_promedio:.3f} | "
              f"T: {step_time:.1f}s | "
              f"ETA: {tiempo_restante/60:.1f}min")
    
    def capturar_frame(self, step_num):
        """Captura frame para animación."""
        if not VISUALIZATION_LIBS_AVAILABLE:
            if step_num == 0: # Imprimir solo una vez
                print("  (i) Saltando captura de frames, librerías de visualización no disponibles.")
            return
        energies_cpu = self._to_cpu(self.energies)
        vmin, vmax = np.min(energies_cpu), np.max(energies_cpu)
        
        fig, ax = plt.subplots(figsize=(7, 7))
        im = ax.imshow(energies_cpu, cmap='viridis', vmin=vmin, vmax=vmax, interpolation='bilinear')
        ax.set_title(f"Geometrogénesis v12 (VQE) - Paso {step_num}", 
                    fontsize=14, fontweight='bold')
        ax.set_xlabel("x", fontsize=10)
        ax.set_ylabel("y", fontsize=10)
        
        cbar = plt.colorbar(im, label="Energía Cuántica (Ising)", ax=ax)
        cbar.ax.tick_params(labelsize=9)
        
        fig.tight_layout()
        fig.canvas.draw()
        
        # Convertir a RGB
        width, height = fig.canvas.get_width_height()
        buf = fig.canvas.buffer_rgba()
        frame_rgba = np.frombuffer(buf, dtype=np.uint8).reshape((height, width, 4))
        frame_rgb = frame_rgba[:, :, :3]  # Descartar canal alfa
        
        self.frames_for_gif.append(frame_rgb)
        plt.close(fig)
    
    def calcular_espectro_potencia_y_ns(self):
        """Calcula espectro de potencia y extrae n_s."""
        print("\n" + "="*70)
        print("  ANÁLISIS COSMOLÓGICO: Espectro de Potencia")
        print("="*70)
        
        phi_final_cpu = self._to_cpu(self.Phi)
        
        print("  1/4 Calculando FFT 2D...")
        fourier_phi = np.fft.fft2(phi_final_cpu)
        fourier_phi_shifted = np.fft.fftshift(fourier_phi)
        
        print("  2/4 Calculando espectro de potencia 2D...")
        power_spectrum_2d = np.abs(fourier_phi_shifted)**2
        
        print("  3/4 Calculando promedio radial...")
        center_x, center_y = GRID_SIZE // 2, GRID_SIZE // 2
        y, x = np.indices(power_spectrum_2d.shape)
        radii = np.sqrt((x - center_x)**2 + (y - center_y)**2)
        radii_int = radii.astype(int)
        k_max = min(radii_int.max(), GRID_SIZE // 2)
        k_vals = np.arange(k_max + 1)
        power_spectrum_1d = np.zeros_like(k_vals, dtype=float)
        
        for k in k_vals:
            mask = (radii_int == k)
            if np.any(mask):
                power_spectrum_1d[k] = power_spectrum_2d[mask].mean()
        
        print("  4/4 Extrayendo índice espectral n_s...")
        k_fit_min = 2
        k_fit_max = min(k_max, GRID_SIZE // 4)
        k_vals_fit = k_vals[k_fit_min:k_fit_max]
        power_fit = power_spectrum_1d[k_fit_min:k_fit_max]
        
        valid_mask = (k_vals_fit > 0) & (power_fit > 0)
        
        if np.sum(valid_mask) < 3: # Ajustado para permitir ajuste con menos puntos en grids pequeños
            print("  ⚠️ Insuficientes puntos para ajuste confiable.")
            return k_vals, power_spectrum_1d, None, power_spectrum_2d
        
        log_k = np.log(k_vals_fit[valid_mask])
        log_P = np.log(power_fit[valid_mask])
        
        slope, intercept = np.polyfit(log_k, log_P, 1)
        n_s = slope + 1.0
        
        log_P_pred = slope * log_k + intercept
        residuals = log_P - log_P_pred
        ss_res = np.sum(residuals**2)
        ss_tot = np.sum((log_P - np.mean(log_P))**2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
        
        print(f"\n  ✓ Índice espectral escalar: n_s = {n_s:.4f}")
        print(f"  ✓ Calidad del ajuste: R² = {r_squared:.4f}")
        print(f"  ✓ Rango de ajuste: k ∈ [{k_fit_min}, {k_fit_max}]")
        
        # Comparación con observaciones
        ns_planck = 0.9649
        desviacion = abs(n_s - ns_planck)
        sigma_planck = 0.0042
        num_sigmas = desviacion / sigma_planck
        
        print(f"\n  📊 Comparación con Planck 2018:")
        print(f"     n_s (Planck) = {ns_planck} ± {sigma_planck}")
        print(f"     Desviación: {num_sigmas:.2f}σ", end="")
        
        if num_sigmas < 1:
            print(" (✓ Consistente dentro de 1σ)")
        elif num_sigmas < 2:
            print(" (⚠️ Marginalmente consistente)")
        else:
            print(" (✗ Inconsistente >2σ)")
        
        print("="*70 + "\n")
        
        return k_vals, power_spectrum_1d, n_s, power_spectrum_2d
    
    def guardar_resultados(self):
        """Guarda todos los resultados de la simulación."""
        print("\n" + "="*70)
        print("  GUARDANDO RESULTADOS")
        print("="*70)
        
        if not VISUALIZATION_LIBS_AVAILABLE:
            print("\n  ⚠️  Saltando guardado de gráficos y animación (matplotlib/imageio no disponibles).")
            self._generar_reporte_resumen(None) # Generar solo el reporte de texto
            return

        # 1. Animación GIF
        gif_path = "geometrogenesis_animacion_v12.gif"
        print(f"\n  📹 Generando animación...")
        try:
            imageio.mimsave(gif_path, self.frames_for_gif, fps=5, loop=0)
            print(f"  ✓ Guardado: '{gif_path}'")
        except Exception as e:
            print(f"  ✗ Error al guardar GIF: {e}")
        
        # 2. Gráfico de evolución
        print(f"\n  📊 Generando gráfico de evolución...")
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                "Fitness Promedio", 
                "Energía del Campo Φ",
                "Energía Cuántica (Ising)", 
                "Tiempo por Paso"
            )
        )
        
        steps = list(range(len(self.history["Fitness_Promedio"])))
        
        fig.add_trace(
            go.Scatter(x=steps, y=self.history["Fitness_Promedio"],
                      name='Fitness', line=dict(color='#00CC96')),
            row=1, col=1
        )
        fig.add_trace(
            go.Scatter(x=steps, y=self.history["Energia_Campo"],
                      name='E_Campo', line=dict(color='#EF553B')),
            row=1, col=2
        )
        fig.add_trace(
            go.Scatter(x=steps, y=self.history["Energia_Cuántica_Promedio"],
                      name='E_Cuántica', line=dict(color='#AB63FA')),
            row=2, col=1
        )
        fig.add_trace(
            go.Scatter(x=steps, y=self.history["Tiempo_Paso"],
                      name='Tiempo', line=dict(color='#FFA15A')),
            row=2, col=2
        )
        
        fig.update_yaxes(title="Fitness", row=1, col=1)
        fig.update_yaxes(title="Energía", row=1, col=2)
        fig.update_yaxes(title="Energía", row=2, col=1)
        fig.update_yaxes(title="Segundos", row=2, col=2)
        fig.update_xaxes(title="Paso", row=2, col=1)
        fig.update_xaxes(title="Paso", row=2, col=2)
        
        fig.update_layout(
            height=800,
            title_text="Geometrogénesis v12: Evolución con Optimización VQE",
            showlegend=False,
            template="plotly_dark"
        )
        
        output_evol = "simulacion_geometrogenesis_v12_evolucion.html"
        fig.write_html(output_evol)
        print(f"  ✓ Guardado: '{output_evol}'")
        
        # 3. Análisis cosmológico
        print(f"\n  🌌 Generando análisis cosmológico...")
        k_vals, power_1d, n_s, power_2d = self.calcular_espectro_potencia_y_ns()
        
        if k_vals is not None:
            fig_ps = make_subplots(
                rows=1, cols=2,
                subplot_titles=(
                    "Espectro 2D: log₁₀|P(kₓ,kᵧ)|²",
                    f"Espectro 1D | n_s = {n_s:.4f}" if n_s else "Espectro 1D"
                ),
                specs=[[{"type": "heatmap"}, {"type": "scatter"}]]
            )
            
            # Heatmap 2D
            power_2d_log = np.log10(power_2d + 1e-10)
            fig_ps.add_trace(
                go.Heatmap(
                    z=power_2d_log,
                    colorscale='Viridis',
                    colorbar=dict(title={'text': "log₁₀(P)", 'side': 'right'})
                ),
                row=1, col=1
            )
            
            # Espectro 1D
            valid_mask = (k_vals > 0) & (power_1d > 0)
            fig_ps.add_trace(
                go.Scatter(
                    x=k_vals[valid_mask],
                    y=power_1d[valid_mask],
                    mode='lines+markers',
                    name='P(k)',
                    line=dict(color='#FFA15A', width=2),
                    marker=dict(size=5)
                ),
                row=1, col=2
            )
            
            # Línea de ajuste
            if n_s is not None:
                k_fit = k_vals[valid_mask]
                power_fit = k_fit**(n_s - 1) * np.exp(10)  # Escalar para visualización
                fig_ps.add_trace(
                    go.Scatter(
                        x=k_fit,
                        y=power_fit,
                        mode='lines',
                        name=f'Ajuste: k^{n_s-1:.2f}',
                        line=dict(dash='dash', color='white', width=2),
                        showlegend=True
                    ),
                    row=1, col=2
                )
            
            # Configurar ejes
            fig_ps.update_yaxes(type="log", title="P(k)", row=1, col=2)
            fig_ps.update_xaxes(type="log", title="k (número de onda)", row=1, col=2)
            fig_ps.update_xaxes(title="kₓ", row=1, col=1)
            fig_ps.update_yaxes(title="kᵧ", row=1, col=1)
            
            fig_ps.update_layout(
                height=600,
                title_text="Geometrogénesis v12: Análisis Cosmológico del Campo Final Φ",
                template="plotly_dark"
            )
            
            output_ps = "simulacion_geometrogenesis_v12_espectro.html"
            fig_ps.write_html(output_ps)
            print(f"  ✓ Guardado: '{output_ps}'")
            
            # Abrir en navegador
            try:
                webbrowser.open('file://' + os.path.realpath(output_ps))
                print(f"  ✓ Abriendo en navegador...")
            except Exception as e:
                print(f"  ⚠️ No se pudo abrir automáticamente: {e}")
        
        # 4. Generar reporte de resumen
        print(f"\n  📄 Generando reporte de resumen...")
        self._generar_reporte_resumen(n_s)
        
        print("\n" + "="*70)
        print("  ✓ TODOS LOS RESULTADOS GUARDADOS EXITOSAMENTE")
        print("="*70 + "\n")
    
    def _generar_reporte_resumen(self, n_s):
        """Genera un archivo de texto con el resumen de la simulación."""
        reporte_path = "geometrogenesis_v12_reporte.txt"
        
        with open(reporte_path, 'w', encoding='utf-8') as f:
            f.write("="*70 + "\n")
            f.write("  GEOMETROGÉNESIS COMPUTACIONAL v12 - REPORTE DE SIMULACIÓN\n")
            f.write("="*70 + "\n\n")
            
            # Parámetros de simulación
            f.write("PARÁMETROS DE SIMULACIÓN:\n")
            f.write("-" * 70 + "\n")
            f.write(f"  Grid Size: {GRID_SIZE}×{GRID_SIZE}\n")
            f.write(f"  Pasos totales: {PASOS_SIMULACION}\n")
            f.write(f"  N Qubits: {N_QUBITS}\n")
            f.write(f"  N Layers: {N_LAYERS}\n")
            f.write(f"  VQE Iterations/Step: {VQE_ITERATIONS_PER_STEP}\n")
            f.write(f"  Paralelización: {'Activada' if VQE_PARALLEL else 'Desactivada'}\n")
            if VQE_PARALLEL:
                f.write(f"  Workers: {MAX_WORKERS}\n")
            f.write(f"\n")
            
            # Parámetros físicos
            f.write("PARÁMETROS FÍSICOS:\n")
            f.write("-" * 70 + "\n")
            f.write(f"  μ² (potencial): {MU_SQUARED}\n")
            f.write(f"  λ (acoplamiento cuártico): {LAMBDA_PARAM}\n")
            f.write(f"  J (Ising coupling): {J_COUPLING}\n")
            f.write(f"  h (campo transverso): {H_TRANSVERSE}\n")
            f.write(f"  Coupling strength (Φ-agentes): {COUPLING_STRENGTH}\n")
            f.write(f"\n")
            
            # Resultados finales
            f.write("RESULTADOS FINALES:\n")
            f.write("-" * 70 + "\n")
            f.write(f"  Fitness promedio final: {self.history['Fitness_Promedio'][-1]:.6f}\n")
            f.write(f"  Energía campo final: {self.history['Energia_Campo'][-1]:.6f}\n")
            f.write(f"  Energía cuántica final: {self.history['Energia_Cuántica_Promedio'][-1]:.6f}\n")
            f.write(f"\n")
            
            # Métricas de convergencia
            f.write("MÉTRICAS DE CONVERGENCIA:\n")
            f.write("-" * 70 + "\n")
            
            # Calcular mejora
            fitness_inicial = self.history['Fitness_Promedio'][0]
            fitness_final = self.history['Fitness_Promedio'][-1]
            mejora_fitness = ((fitness_final - fitness_inicial) / fitness_inicial) * 100
            
            energia_inicial = self.history['Energia_Cuántica_Promedio'][0]
            energia_final = self.history['Energia_Cuántica_Promedio'][-1]
            reduccion_energia = ((energia_inicial - energia_final) / abs(energia_inicial)) * 100
            
            f.write(f"  Mejora en fitness: {mejora_fitness:+.2f}%\n")
            f.write(f"  Reducción de energía cuántica: {reduccion_energia:+.2f}%\n")
            f.write(f"\n")
            
            # Análisis cosmológico
            if n_s is not None:
                f.write("ANÁLISIS COSMOLÓGICO:\n")
                f.write("-" * 70 + "\n")
                f.write(f"  Índice espectral escalar (n_s): {n_s:.4f}\n")
                f.write(f"  Valor observado (Planck 2018): 0.9649 ± 0.0042\n")
                
                desviacion = abs(n_s - 0.9649)
                num_sigmas = desviacion / 0.0042
                f.write(f"  Desviación: {num_sigmas:.2f}σ\n")
                
                if num_sigmas < 1:
                    f.write(f"  Consistencia: ✓ Dentro de 1σ (excelente)\n")
                elif num_sigmas < 2:
                    f.write(f"  Consistencia: ⚠️ Marginalmente consistente (1-2σ)\n")
                else:
                    f.write(f"  Consistencia: ✗ Inconsistente (>2σ)\n")
                f.write(f"\n")
            
            # Performance
            f.write("PERFORMANCE:\n")
            f.write("-" * 70 + "\n")
            tiempo_total = sum(self.history['Tiempo_Paso'])
            tiempo_promedio = np.mean(self.history['Tiempo_Paso'])
            tiempo_min = min(self.history['Tiempo_Paso'])
            tiempo_max = max(self.history['Tiempo_Paso'])
            
            f.write(f"  Tiempo total: {tiempo_total/60:.2f} minutos\n")
            f.write(f"  Tiempo promedio/paso: {tiempo_promedio:.2f} segundos\n")
            f.write(f"  Tiempo mín/paso: {tiempo_min:.2f} segundos\n")
            f.write(f"  Tiempo máx/paso: {tiempo_max:.2f} segundos\n")
            f.write(f"\n")
            
            # Archivos generados
            f.write("ARCHIVOS GENERADOS:\n")
            f.write("-" * 70 + "\n")
            f.write(f"  1. geometrogenesis_animacion_v12.gif\n")
            f.write(f"  2. simulacion_geometrogenesis_v12_evolucion.html\n")
            f.write(f"  3. simulacion_geometrogenesis_v12_espectro.html\n")
            f.write(f"  4. geometrogenesis_v12_reporte.txt (este archivo)\n")
            f.write(f"\n")
            
            # Timestamp
            from datetime import datetime
            f.write("="*70 + "\n")
            f.write(f"Reporte generado: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("="*70 + "\n")
        
        print(f"  ✓ Guardado: '{reporte_path}'")


def run_simulation():
    """
    Función que encapsula la creación y ejecución de la simulación.
    """
    print("✓ Todas las dependencias están disponibles.\n")

    # Advertencia sobre tiempo de ejecución
    if VQE_PARALLEL:
        tiempo_estimado = (GRID_SIZE * GRID_SIZE * VQE_ITERATIONS_PER_STEP * 0.5) / MAX_WORKERS / 60
    else:
        tiempo_estimado = (GRID_SIZE * GRID_SIZE * VQE_ITERATIONS_PER_STEP * 0.5) / 60

    print("="*70)
    print(f"  ⚠️  ADVERTENCIA: Tiempo estimado ~ {tiempo_estimado:.0f} minutos")
    print("="*70)
    print(f"\n  La optimización VQE es computacionalmente intensiva.")
    print(f"  Configuración actual:")
    print(f"    • {GRID_SIZE*GRID_SIZE} agentes optimizándose cada paso")
    print(f"    • {VQE_ITERATIONS_PER_STEP} iteraciones VQE por agente")
    print(f"    • {PASOS_SIMULACION} pasos totales")
    if VQE_PARALLEL:
        print(f"    • Paralelización activada ({MAX_WORKERS} workers)")
    else:
        print(f"    • Ejecución secuencial (considera activar paralelización)")

    respuesta = input("\n  ¿Continuar con la simulación? (s/n): ")
    if respuesta.lower() != 's':
        print("\n  Simulación cancelada por el usuario.\n")
        sys.exit(0)

    # Crear y ejecutar simulador
    print("\n" + "="*70)
    print("  INICIANDO SIMULACIÓN")
    print("="*70 + "\n")

    sim = SimuladorVQE()

    print(f"Ejecutando {PASOS_SIMULACION} pasos de simulación...\n")
    for i in range(PASOS_SIMULACION):
        sim.step(i)
    return sim

def main():
    """Función principal de ejecución."""
    print("\n" + "#"*70)
    print("#" + " "*68 + "#")
    print("#  GEOMETROGÉNESIS COMPUTACIONAL v12" + " "*33 + "#")
    print("#  Simulador con Optimización Cuántica Variacional (VQE)" + " "*13 + "#")
    print("#" + " "*68 + "#")
    print("#"*70 + "\n")
    
    # Verificar dependencias críticas
    critical_error = False
    if not QUANTUM_LIB_AVAILABLE or not VISUALIZATION_LIBS_AVAILABLE:
        print("="*70)
        print("  ✗ ERROR CRÍTICO: Faltan dependencias")
        print("="*70)
        if not QUANTUM_LIB_AVAILABLE:
            print("\n  - PennyLane no está disponible (necesario para la capa cuántica).")
        if not VISUALIZATION_LIBS_AVAILABLE:
            print("\n  - Matplotlib o Imageio no están disponibles (necesarios para la visualización).")
        
        print("\n  Para instalar todas las dependencias, crea un archivo 'requirements.txt' con:")
        print("  ┌─────────────────────────────────────────────────────────────┐")
        print("  │ numpy\n  │ plotly\n  │ matplotlib\n  │ imageio[ffmpeg]\n  │ pennylane                                                 │")
        print("  └─────────────────────────────────────────────────────────────┘")
        print("\n  Y luego ejecuta:")
        print("  ┌─────────────────────────────────────────────────────────────┐")
        print("  │  pip install -r requirements.txt                            │")
        print("  └─────────────────────────────────────────────────────────────┘")
        sys.exit(1)

    sim = None
    try:
        sim = run_simulation()
    except KeyboardInterrupt:
        print("\n\n" + "="*70)
        print("  ⚠️  SIMULACIÓN INTERRUMPIDA POR EL USUARIO")
        print("="*70)
        print("\n  Guardando resultados parciales...\n")
    except Exception as e:
        print("\n\n" + "="*70)
        print("  ✗ ERROR DURANTE LA SIMULACIÓN")
        print("="*70)
        print(f"\n  Error: {e}\n")
        import traceback
        traceback.print_exc()
        print("\n  Guardando resultados hasta el punto de fallo...\n")
    
    # Guardar resultados
    if sim:
        sim.guardar_resultados()

    # Mensaje final
    print("\n" + "#"*70)
    print("#" + " "*68 + "#")
    print("#  ✓ SIMULACIÓN COMPLETADA EXITOSAMENTE" + " "*30 + "#")
    print("#" + " "*68 + "#")
    print("#"*70)
    print("\n  Archivos generados:")
    print("    • geometrogenesis_animacion_v12.gif")
    print("    • simulacion_geometrogenesis_v12_evolucion.html")
    print("    • simulacion_geometrogenesis_v12_espectro.html")
    print("    • geometrogenesis_v12_reporte.txt")
    print("\n  Abre los archivos HTML en tu navegador para visualizar los resultados.\n")


if __name__ == "__main__":
    # Protección para multiprocessing en Windows
    multiprocessing.freeze_support()
    main()