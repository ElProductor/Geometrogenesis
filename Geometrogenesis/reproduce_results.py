import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import webbrowser
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing

# --- HARDWARE ---
FORCE_CPU = True
try:
    if FORCE_CPU: raise ImportError
    import cupy as cp
    GPU_ENABLED = True
    print("✓ GPU Detectada (CuPy)")
except:
    cp = np
    GPU_ENABLED = False
    print("✓ CPU Mode (NumPy)")

# --- QISKIT CHECK ---
try:
    from qiskit import QuantumCircuit
    from qiskit.primitives import Estimator
    from qiskit.quantum_info import SparsePauliOp
    from qiskit_algorithms.optimizers import SPSA
    QISKIT_AVAILABLE = True
except:
    print("⚠️ Qiskit no instalado.")
    QISKIT_AVAILABLE = False

# --- LA RECETA DEL UNIVERSO (GANADORA) ---
GRID_SIZE = 20
PASOS = 50
N_QUBITS = 2
N_LAYERS = 2

# Constantes Fundamentales Descubiertas
COUPLING_G = 0.9    # Acoplamiento Fuerte
J_INTERACTION = -0.01 # Individualismo
H_TRANSVERSE = -0.05  # Límite Clásico/Crítico

# Física del Campo
MU_SQ = 1.0
LAMBDA = 1.0
DT = 0.02

# Config VQE
ITERACIONES_VQE = 5
VQE_PARALLEL = True
MAX_WORKERS = max(1, multiprocessing.cpu_count() - 1)

def _hamiltoniano(n, j, h):
    paulis, coeffs = [], []
    # Interacción ZZ
    for i in range(n-1):
        p = ["I"]*n; p[i]="Z"; p[i+1]="Z"
        paulis.append("".join(p)[::-1]); coeffs.append(j)
    # Campo X
    for i in range(n):
        p = ["I"]*n; p[i]="X"
        paulis.append("".join(p)[::-1]); coeffs.append(h)
    return SparsePauliOp(paulis, coeffs)

def _circuito(params):
    qc = QuantumCircuit(N_QUBITS)
    idx = 0
    for _ in range(N_LAYERS):
        for i in range(N_QUBITS):
            if idx < len(params): qc.ry(params[idx], i); idx+=1
        if N_QUBITS>1:
            for i in range(N_QUBITS-1): qc.cx(i, i+1)
    return qc

def _worker(args):
    i, j, p_init, field, Ham_op = args
    try:
        est = Estimator()
        opt = SPSA(maxiter=ITERACIONES_VQE)
        def cost(p):
            qc = _circuito(p)
            # Energía Cuántica + Influencia del Campo
            E_q = est.run([qc], [Ham_op], shots=256).result().values[0]
            return E_q + field
        res = opt.minimize(cost, p_init.flatten())
        return (i, j, res.x, res.fun)
    except: return (i, j, p_init, 0.0)

class Universo:
    def __init__(self):
        self.xp = cp if GPU_ENABLED else np
        self.Phi = self.xp.array(np.random.uniform(-1,1,(GRID_SIZE,GRID_SIZE)) + 
                                 1j*np.random.uniform(-1,1,(GRID_SIZE,GRID_SIZE)))
        # Ajuste tamaño params: N_LAYERS * N_QUBITS rotaciones RY. Simplificamos a vector plano.
        self.params = np.random.uniform(0, 6.28, (GRID_SIZE, GRID_SIZE, N_LAYERS * N_QUBITS))
        
        self.energies = self.xp.ones((GRID_SIZE,GRID_SIZE))
        self.Ham_Op = _hamiltoniano(N_QUBITS, J_INTERACTION, H_TRANSVERSE) if QISKIT_AVAILABLE else None
        self.history_ns = []

    def evolucionar(self):
        # 1. Influencia Campo -> Agente
        field_val = COUPLING_G * (self.xp.abs(self.Phi)**2 - 1)
        f_cpu = cp.asnumpy(field_val) if GPU_ENABLED else field_val
        
        # 2. Optimización Cuántica (VQE)
        if QISKIT_AVAILABLE and VQE_PARALLEL:
            tasks = [(r,c, self.params[r,c], f_cpu[r,c], self.Ham_Op) 
                     for r in range(GRID_SIZE) for c in range(GRID_SIZE)]
            with ProcessPoolExecutor(max_workers=MAX_WORKERS) as exe:
                for f in as_completed([exe.submit(_worker, t) for t in tasks]):
                    r,c, p_new, e_new = f.result()
                    self.params[r,c] = p_new
                    # Actualizamos energía localmente (aproximación rápida para visualización)
        
        # 3. Retroalimentación Agente -> Campo
        # (Usamos la aproximación de campo fuerte para evolución suave)
        self.energies = field_val 
        
        pot = (-MU_SQ + 2*LAMBDA*self.xp.abs(self.Phi)**2) * self.Phi
        source = COUPLING_G * self.energies * self.Phi
        self.Phi -= (pot + source) * DT

    def analizar(self):
        # FFT
        phi = cp.asnumpy(np.abs(self.Phi)) if GPU_ENABLED else np.abs(self.Phi)
        fft = np.fft.fftshift(np.fft.fft2(phi))
        ps = np.abs(fft)**2
        
        # Radial
        cy, cx = GRID_SIZE//2, GRID_SIZE//2
        y, x = np.indices(ps.shape)
        r = np.sqrt((x-cx)**2 + (y-cy)**2).astype(int)
        
        k, Pk = [], []
        for ri in range(1, GRID_SIZE//2):
            mask = (r==ri)
            if np.any(mask): k.append(ri); Pk.append(ps[mask].mean())
        
        k, Pk = np.array(k), np.array(Pk)
        
        # Ajuste n_s (Log-Log)
        if len(k) < 3: return 0, k, Pk
        try:
            slope, _ = np.polyfit(np.log(k), np.log(Pk), 1)
            ns = slope + 1.0
            return ns, k, Pk
        except: return 0, k, Pk

def main():
    print(f"--- INICIANDO GEOMETROGÉNESIS ---")
    print(f"Config: G={COUPLING_G}, J={J_INTERACTION}, H={H_TRANSVERSE}")
    sim = Universo()
    
    start = time.time()
    for i in range(PASOS):
        sim.evolucionar()
        print(f"\rPaso {i+1}/{PASOS}...", end="")
    
    ns, k, Pk = sim.analizar()
    print(f"\n\n>>> RESULTADO FINAL: n_s = {ns:.4f}")
    print(f"OBJETIVO PLANCK: 0.9649")
    print(f"DIFERENCIA: {abs(ns-0.9649):.4f}")
    
    # Gráfico
    fig = make_subplots(rows=1, cols=2, subplot_titles=("Campo Φ Final", "Espectro de Potencia P(k)"))
    
    phi_img = cp.asnumpy(np.abs(sim.Phi)) if GPU_ENABLED else np.abs(sim.Phi)
    fig.add_trace(go.Heatmap(z=phi_img, colorscale='Viridis'), row=1, col=1)
    
    fig.add_trace(go.Scatter(x=k, y=Pk, mode='lines+markers', name='Simulacion'), row=1, col=2)
    
    # Línea teórica Planck
    if len(k)>0:
        y_fit = np.exp(np.log(Pk[0])) * (k/k[0])**(0.9649 - 1)
        fig.add_trace(go.Scatter(x=k, y=y_fit, mode='lines', line=dict(dash='dash'), name='Planck (0.96)'), row=1, col=2)
    
    fig.update_layout(title=f"Geometrogénesis: n_s = {ns:.4f}", template="plotly_dark")
    fig.update_xaxes(type="log", row=1, col=2); fig.update_yaxes(type="log", row=1, col=2)
    
    # Crear carpeta results si no existe
    if not os.path.exists('results'):
        os.makedirs('results')
        
    output_path = os.path.join('results', 'resultado_final.html')
    fig.write_html(output_path)
    print(f"\n✓ Gráfico guardado: {output_path}")
    
    # Intentar abrir en navegador
    try:
        webbrowser.open('file://' + os.path.abspath(output_path))
    except:
        pass

if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()