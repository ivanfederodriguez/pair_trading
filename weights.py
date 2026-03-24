import numpy as np

def _normalize(w_raw):
    """Normaliza los pesos para que sumen 1 (100% del portafolio) por cada día."""
    s = w_raw.sum(axis=0)
    s_safe = np.where(s == 0, 1, s)  # Evita división por cero
    return w_raw / s_safe

def equal_weight(positions):
    w_raw = np.abs(positions)
    return _normalize(w_raw)

def risk_parity_weight(positions, spread_vols):
    w_raw = np.abs(positions) / (spread_vols[:, None] + 1e-8)
    return _normalize(w_raw)

def zscore_pure_weight(positions, z_matrix):
    w_raw = np.abs(positions * z_matrix)
    return _normalize(w_raw)

def zscore_squashed_weight(positions, z_matrix, spread_vols):
    z_squashed = np.tanh(np.abs(z_matrix) / 2.0)
    w_raw = (np.abs(positions) * z_squashed) / (spread_vols[:, None] + 1e-8)
    return _normalize(w_raw)

def crossings_weight(positions, z_matrix, spread_vols, crossings_rate):
    w_raw = (np.abs(positions * z_matrix) * crossings_rate[:, None]) / (spread_vols[:, None] + 1e-8)
    return _normalize(w_raw)

def kelly_dynamic_weight(positions, z_matrix, spread_vols):
    # n dinámico: cantidad de activos con posición abierta en cada día específico
    n_dinamico = (np.abs(positions) > 0).sum(axis=0)
    n_dinamico_safe = np.where(n_dinamico == 0, 1, n_dinamico)
    
    # Fórmula: w = |-z / (2 * sigma * n)|
    w_raw = (np.abs(z_matrix) / (2 * spread_vols[:, None] * n_dinamico_safe)) * np.abs(positions)
    return _normalize(w_raw)

def apply_hold_period(w_matrix, positions, half_lives):
    """
    Filtro general: Recibe una matriz de pesos calculada por cualquier estrategia
    y congela el capital asignado a cada posición hasta que supere su Half-Life.
    Una vez superado, permite que el peso vuelva a actualizarse dinámicamente.
    """
    w_held = np.zeros_like(w_matrix)
    n_pairs, n_days = w_matrix.shape
    
    for i in range(n_pairs):
        hl_val = half_lives[i]
        
        # Validar el half-life (si es inválido o muy corto, mínimo 1 día)
        if np.isnan(hl_val) or np.isinf(hl_val) or hl_val < 1:
            hl = 1
        else:
            hl = int(round(hl_val))
            
        dias_abierto = 0
        peso_fijado = 0.0
        
        for t in range(n_days):
            if positions[i, t] != 0: # Si la posición está viva
                
                # Si recién abre (día 0) o si ya superó el 'n' estimado:
                # permitimos que el peso tome el valor actual de la estrategia
                if dias_abierto == 0 or dias_abierto >= hl:
                    peso_fijado = w_matrix[i, t]
                
                # Asignamos el peso (que puede estar congelado o recién actualizado)
                w_held[i, t] = peso_fijado
                dias_abierto += 1
            else:
                # Si la posición está cerrada, reseteamos el contador
                dias_abierto = 0
                w_held[i, t] = 0.0
                
    # Medida de seguridad: Si al congelar pesos antiguos la suma total de capital
    # activo ese día supera el 100% (1.0), normalizamos proporcionalmente para no apalancarse.
    suma_diaria = w_held.sum(axis=0)
    exceso = suma_diaria > 1.0
    w_held[:, exceso] = w_held[:, exceso] / suma_diaria[exceso]
    
    return w_held