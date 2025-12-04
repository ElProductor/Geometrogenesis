# Geometrogénesis

**Protocolo de calibración de un universo simulado mediante computación cuántica.**

## Objetivo

El objetivo del proyecto Geometrogénesis es simular un universo simplificado y calibrar sus parámetros fundamentales para que coincidan con valores observados o teóricos. En esta implementación, utilizamos el framework de computación cuántica **Qiskit** para encontrar una configuración de parámetros que produzca un "índice espectral escalar" ($n_s$) objetivo de **0.9649**.

Este valor está inspirado en las mediciones cosmológicas del fondo cósmico de microondas (CMB), como las obtenidas por el satélite Planck.

## El Protocolo

El protocolo se divide en varias fases de calibración, donde cada una ajusta un parámetro específico del Hamiltoniano del sistema cuántico:

1.  **Fase 1 (No incluida en el script):** Se busca un valor aproximado para el `COUPLING_STRENGTH` (acoplamiento).
2.  **Fase 2: Barrido de Interacción Cuántica (`J_COUPLING`):** Con el acoplamiento fijado, se realiza un ajuste fino del estado de fase del qubit mediante una compuerta `RZ`.
3.  **Fase 3: Barrido de Campo Transverso (`H_TRANSVERSE`):** Se introduce un "caos cuántico" mediante una compuerta `RX` para explorar otras regiones del espacio de Hilbert.
4.  **Fase 5: El Límite Clásico:** Se exploran valores de `H_TRANSVERSE` muy cercanos a cero para una calibración de alta precisión, buscando alcanzar el objetivo final.

## Cómo ejecutar

1.  Asegúrate de tener Python y las librerías necesarias instaladas:
    ```bash
    pip install qiskit qiskit-aer matplotlib numpy
    ```
2.  Abre el archivo `motor_de_valores.py`.
3.  Modifica la variable `FASE_A_EJECUTAR` para seleccionar la fase del protocolo que deseas simular (ej. `FASE_A_EJECUTAR = 5`).
4.  Ejecuta el script:
    ```bash
    python motor_de_valores.py
    ```