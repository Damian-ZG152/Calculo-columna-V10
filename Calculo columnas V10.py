"""
=========================================================================================
 🏗️ APP DE DISEÑO DE COLUMNAS NTC 2023 
    (TABLAS RESUMEN, PUNTO CONVENIENTE EXACTO Y GRÁFICAS SIN COLISIÓN)
=========================================================================================
"""

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Áreas nominales de varillas comunes (cm2)
AREAS_VARILLAS = {
    2: 0.3167, 2.5: 0.4948, 3: 0.7126, 4: 1.2668,
    5: 1.9793, 6: 2.8502, 7: 3.8795, 8: 5.0671,
    9: 6.4130, 10: 7.9173, 11: 9.5800, 12: 11.4009
}

# ============================================================================
# MOTOR MATEMÁTICO ESTRUCTURAL
# ============================================================================
class Bloque:
    def __init__(self, b, h, y_top):
        self.b = b
        self.h = h
        self.y_top = y_top
        self.y_bot = y_top + h

class Lecho:
    def __init__(self, As, d, cant=2):
        self.As = As
        self.d = d
        self.cant = cant

class SeccionColumna:
    def __init__(self, fc, fy, bloques, lechos):
        self.fc = fc
        self.fy = fy
        self.fcc = 0.85 * fc
        
        # Factor Beta 1 según NTC 2023
        if fc <= 280:
            self.beta1 = 0.85
        else:
            self.beta1 = max(0.65, 1.05 - (fc / 1400.0))
            
        self.Es = 2000000.0
        self.ecu = 0.003
        self.ey = self.fy / self.Es
        
        self.bloques = bloques
        self.lechos = sorted(lechos, key=lambda x: x.d)
        
        self.H = max(b.y_bot for b in self.bloques) if self.bloques else 0
        self.Ag = sum(b.b * b.h for b in self.bloques)
        self.As_total = sum(l.As for l in self.lechos)
        
        self._calcular_centroide_plastico()

    def _calcular_centroide_plastico(self):
        """ Calcula Po y Yp midiendo desde la fibra superior de compresión """
        sum_F = 0
        sum_Fy = 0
        self.detalles_yp = []
        
        for i, b in enumerate(self.bloques):
            F_c = self.fcc * (b.b * b.h)
            y_cg = b.y_top + b.h / 2.0
            sum_F += F_c
            sum_Fy += F_c * y_cg
            self.detalles_yp.append({
                "Elemento": f"Conc. B{i+1}", "Área (cm²)": b.b * b.h, 
                "Esfuerzo": self.fcc, "Fuerza (kg)": F_c, 
                "Brazo Y (cm)": y_cg, "Momento (kg-cm)": F_c * y_cg
            })
            
        for i, l in enumerate(self.lechos):
            # En Yp todo fluye a compresión. Se descuenta el concreto desplazado.
            F_s = l.As * (self.fy - self.fcc)
            sum_F += F_s
            sum_Fy += F_s * l.d
            self.detalles_yp.append({
                "Elemento": f"Acero L{i+1}", "Área (cm²)": l.As, 
                "Esfuerzo": self.fy - self.fcc, "Fuerza (kg)": F_s, 
                "Brazo Y (cm)": l.d, "Momento (kg-cm)": F_s * l.d
            })
            
        self.Yp = sum_Fy / sum_F if sum_F != 0 else self.H / 2.0
        self.Po = sum_F

    def _obtener_d_conveniente(self, lechos_calc):
        """ Encuentra el valor c=d_i que anule un lecho intermedio de acero """
        ds = sorted(list(set([l.d for l in lechos_calc])))
        if len(ds) <= 2:
            return sum(ds)/len(ds) if ds else self.H/2.0
        
        # Filtramos los lechos extremos para buscar sólo los intermedios
        middle_ds = ds[1:-1]
        if not middle_ds: return self.H / 2.0
        
        # Tomar el lecho intermedio más cercano al centro de la sección
        return min(middle_ds, key=lambda x: abs(x - self.H/2.0))

    def evaluar_estado(self, c=None, invertido=False, tipo="flexion", punto_nombre=""):
        """ Evalúa fuerzas y momentos para un eje neutro dado o casos teóricos puros """
        if invertido:
            bloques_calc = [Bloque(b.b, b.h, self.H - b.y_bot) for b in self.bloques]
            lechos_calc = [Lecho(l.As, self.H - l.d, l.cant) for l in self.lechos]
            Yp_calc = self.H - self.Yp
        else:
            bloques_calc = self.bloques
            lechos_calc = self.lechos
            Yp_calc = self.Yp

        if tipo == "compresion_pura":
            c_val = float('inf')
            a = self.H
        elif tipo == "tension_pura":
            c_val = 0.0
            a = 0.0
        else:
            c_val = c
            a = min(self.beta1 * c_val, self.H)

        Pn_c, Mn_c = 0.0, 0.0
        det_c = []
        
        # Concreto: En tensión pura teórica el concreto aporta 0 kg de resistencia
        if tipo != "tension_pura":
            for i, b in enumerate(bloques_calc):
                y_sup = max(b.y_top, 0)
                y_inf = min(b.y_bot, a)
                
                if y_inf > y_sup:
                    area_comp = b.b * (y_inf - y_sup)
                    F_c = area_comp * self.fcc
                    y_cg = (y_sup + y_inf) / 2.0
                    brazo = Yp_calc - y_cg
                    Pn_c += F_c
                    Mn_c += F_c * brazo
                    det_c.append({'Elemento': f'Bloque {i+1}', 'Área (cm²)': area_comp, 'Fuerza (kg)': F_c, 'Brazo a Yp (cm)': brazo, 'Momento (kg-cm)': F_c * brazo})

        Pn_s, Mn_s, et = 0.0, 0.0, 0.0
        det_s = []
        d_t = max([l.d for l in lechos_calc]) if lechos_calc else 0

        # Acero
        for i, l in enumerate(lechos_calc):
            d = l.d
            if tipo == "compresion_pura":
                fsi = self.fy
                esi = self.ey
            elif tipo == "tension_pura":
                fsi = -self.fy
                esi = -self.ey
            else:
                esi = self.ecu * (c_val - d) / c_val if c_val > 0 else -self.ey
                fsi = max(-self.fy, min(self.fy, esi * self.Es))

            if d == d_t: et = -esi

            # Descuento estricto del concreto desplazado (solo si está a compresión y dentro del bloque a)
            if tipo != "tension_pura" and d <= a and fsi > 0:
                fsi_neto = fsi - self.fcc
            else:
                fsi_neto = fsi

            F_s = l.As * fsi_neto
            brazo = Yp_calc - d
            Pn_s += F_s
            Mn_s += F_s * brazo
            
            y_real = self.H - l.d if invertido else l.d
            det_s.append({'Elemento': f'Lecho {i+1} (y={y_real:.1f})', 'Def unit. (ε)': esi, 'fs neto (kg/cm²)': fsi_neto, 'Fuerza (kg)': F_s, 'Brazo a Yp (cm)': brazo, 'Momento (kg-cm)': F_s * brazo})

        Pn = Pn_c + Pn_s
        Mn = Mn_c + Mn_s
        if invertido: Mn = -Mn

        # Factores de Resistencia FR (NTC 2023)
        if tipo == "compresion_pura": FR = 0.65
        elif tipo == "tension_pura": FR = 0.90
        else:
            if et <= self.ey: FR = 0.65
            elif et >= self.ey + 0.003: FR = 0.90
            else: FR = 0.65 + 0.25 * ((et - self.ey) / 0.003)

        return {
            'nombre': punto_nombre, 'c': c_val, 'a': a, 'et': et,
            'Pn': Pn/1000.0, 'Mn': Mn/100000.0, 'FR': FR,
            'Pu': (Pn*FR)/1000.0, 'Mu': (Mn*FR)/100000.0,
            'det_c': det_c, 'det_s': det_s, 'invertido': invertido
        }

    def obtener_puntos_clave(self, invertido=False):
        lechos_calc = [Lecho(l.As, self.H - l.d, l.cant) for l in self.lechos] if invertido else self.lechos
        d_t = max([l.d for l in lechos_calc]) if lechos_calc else self.H
        
        c_bal = (0.003 / (0.003 + self.ey)) * d_t
        c_tra = (0.003 / (0.003 + 0.0051)) * d_t
        
        # EL PUNTO CONVENIENTE INTELIGENTE: c = d_i
        c_conv = self._obtener_d_conveniente(lechos_calc)
        
        prefijo = "(-)" if invertido else "(+)"
        
        pts_intermedios = [
            self.evaluar_estado(c=c_bal, invertido=invertido, punto_nombre=f"Falla Balanceada"),
            self.evaluar_estado(c=c_conv, invertido=invertido, punto_nombre=f"Pto Conveniente (c=d)"),
            self.evaluar_estado(c=c_tra, invertido=invertido, punto_nombre=f"Zona Transición")
        ]
        
        # Ordenamos los puntos intermedios por Carga Axial descendente para mantener la lógica geométrica
        pts_intermedios = sorted(pts_intermedios, key=lambda x: x['Pn'], reverse=True)
        
        return [
            self.evaluar_estado(tipo="compresion_pura", invertido=invertido, punto_nombre=f"Compresión Pura")
        ] + pts_intermedios + [
            self.evaluar_estado(tipo="tension_pura", invertido=invertido, punto_nombre=f"Tensión Pura")
        ]

    def obtener_lista_puntos_clave(self):
        """ Combina puntos positivos y negativos y les asigna un ID numérico secuencial """
        pts_pos = self.obtener_puntos_clave(False)
        pts_neg = self.obtener_puntos_clave(True)
        
        lista = []
        idx = 1
        
        # 1. Compresión Pura
        p_comp = next(p for p in pts_pos if "Compresión Pura" in p['nombre'])
        p_comp['id'] = idx; idx+=1; lista.append(p_comp)
        
        # 2, 3, 4. Puntos Positivos
        for p in pts_pos:
            if "Compresión" not in p['nombre'] and "Tensión" not in p['nombre']:
                p['id'] = idx; idx+=1; lista.append(p)
                
        # 5. Tensión Pura
        p_tens = next(p for p in pts_pos if "Tensión Pura" in p['nombre'])
        p_tens['id'] = idx; idx+=1; lista.append(p_tens)
        
        # 6, 7, 8. Puntos Negativos (ordenados por Pn para subir por el lado izquierdo del diagrama)
        neg_filtered = [p for p in pts_neg if "Compresión" not in p['nombre'] and "Tensión" not in p['nombre']]
        neg_filtered.sort(key=lambda x: x['Pn']) 
        for p in neg_filtered:
            p['id'] = idx; idx+=1; lista.append(p)
            
        return lista

    def obtener_diagrama_completo(self):
        d_t_pos = max([l.d for l in self.lechos]) if self.lechos else self.H
        d_t_neg = max([self.H - l.d for l in self.lechos]) if self.lechos else self.H
        
        c_pos = np.concatenate([np.linspace(self.H*3, self.H, 15), np.linspace(self.H, d_t_pos, 25), np.linspace(d_t_pos, 0.1, 40)])
        c_neg = np.concatenate([np.linspace(0.1, d_t_neg, 40), np.linspace(d_t_neg, self.H, 25), np.linspace(self.H, self.H*3, 15)])
        
        p_comp = self.evaluar_estado(tipo="compresion_pura", invertido=False)
        p_tens = self.evaluar_estado(tipo="tension_pura", invertido=False)
        
        curva_pos = [self.evaluar_estado(c, invertido=False) for c in c_pos]
        curva_neg = [self.evaluar_estado(c, invertido=True) for c in c_neg]
        
        return [p_comp] + curva_pos + [p_tens] + curva_neg + [p_comp]

# ============================================================================
# FUNCIONES DE RENDERIZADO VISUAL, TABLAS Y GRÁFICOS ANTI-COLISIÓN
# ============================================================================
def generar_tabla_resumen(pts):
    datos = []
    for p in pts:
        c_str = "∞" if p['c'] == float('inf') else ("0.00" if p['c'] == 0.0 else f"{p['c']:.2f}")
        datos.append({
            "ID": str(p['id']),
            "Condición / Punto": p['nombre'].replace(" (+)", "").replace(" (-)", ""),
            "c (cm)": c_str,
            "Mn (ton-m)": p['Mn'],
            "Pn (ton)": p['Pn'],
            "Factor FR": p['FR'],
            "Mu (ton-m)": p['Mu'],
            "Pu (ton)": p['Pu']
        })
    return pd.DataFrame(datos)

def plot_diagrama_con_flechas(sec, titulo):
    puntos_curva = sec.obtener_diagrama_completo()
    Mn = [p['Mn'] for p in puntos_curva]
    Pn = [p['Pn'] for p in puntos_curva]
    Mu = [p['Mu'] for p in puntos_curva]
    Pu = [p['Pu'] for p in puntos_curva]
    
    fig, ax = plt.subplots(figsize=(10, 8), dpi=120)
    ax.plot(Mn, Pn, color='#7f8c8d', linestyle='--', lw=1.5, label="Curva Nominal (NTC)")
    ax.plot(Mu, Pu, color='#000080', lw=2.5, label="Curva de Diseño (FR)")
    ax.fill_between(Mu, Pu, 0, color='#e0f2f7', alpha=0.6)
    
    # Extraer los puntos clave
    pts_pos = sec.obtener_puntos_clave(invertido=False)
    pts_neg = sec.obtener_puntos_clave(invertido=True)
    
    # Separamos Tensión y Compresión (que son únicos y van centrados)
    pt_comp = pts_pos[0]
    pt_tens = pts_pos[-1]
    
    # Extraemos los 3 puntos intermedios de cada cara
    mid_pos = sorted(pts_pos[1:-1], key=lambda x: x['Pn'], reverse=True)
    mid_neg = sorted(pts_neg[1:-1], key=lambda x: x['Pn'], reverse=True)
    
    max_M, min_M = max(max(Mn), 10), min(min(Mn), -10)
    max_P, min_P = max(max(Pn), 10), min(min(Pn), -10)
    
    bbox_style = dict(boxstyle="round,pad=0.4", fc="#ffffff", ec="#bdc3c7", lw=1.2, alpha=0.95)
    arrow_style = dict(arrowstyle="-|>", color="#34495e", lw=1.5, shrinkB=5)
    
    # Obtenemos la lista con IDs para la gráfica
    pts_todos = sec.obtener_lista_puntos_clave()

    def anotar_punto(pt, x_target, y_target, ha, va):
        m_nom, p_nom = pt['Mn'], pt['Pn']
        m_dis, p_dis = pt['Mu'], pt['Pu']
        
        # Encontramos el ID correspondiente
        pt_id = next((p['id'] for p in pts_todos if p['c'] == pt['c'] and p['invertido'] == pt['invertido'] and p['nombre'] == pt['nombre']), "")
        
        # Dibujamos el punto en la curva
        ax.plot(m_nom, p_nom, 'o', color='red', markersize=6, zorder=5)
        ax.plot(m_dis, p_dis, 'o', color='blue', markersize=5, zorder=5)
        
        # Etiqueta del ID al lado del punto nominal
        ax.annotate(str(pt_id), (m_nom, p_nom), xytext=(0, 8), textcoords="offset points",
                    ha='center', va='bottom', fontsize=10, fontweight='bold', color='darkred',
                    bbox=dict(boxstyle="circle,pad=0.15", fc="#f1c40f", ec="none", alpha=0.9))
        
        nombre_corto = pt['nombre'].replace(" (+)", "").replace(" (-)", "")
        c_str = "∞" if pt['c'] == float('inf') else f"{pt['c']:.1f}"
        texto = f"[{pt_id}] {nombre_corto}\nc={c_str}\nNom: ({m_nom:.1f}, {p_nom:.1f})\nDis: ({m_dis:.1f}, {p_dis:.1f})"
        
        # Conectamos con una flecha desde la etiqueta hasta el punto exacto usando 'data' coords
        ax.annotate(texto, xy=(m_nom, p_nom), xytext=(x_target, y_target), textcoords="data", 
                    ha=ha, va=va, fontsize=8, color="#2c3e50", fontweight='bold',
                    arrowprops=arrow_style, bbox=bbox_style, zorder=10)

    # Anotar Compresión Pura y Tensión Pura en el centro
    anotar_punto(pt_comp, 0, max_P + (max_P - min_P)*0.10, 'center', 'bottom')
    anotar_punto(pt_tens, 0, min_P - (max_P - min_P)*0.10, 'center', 'top')
    
    # 📌 ESTRATEGIA DE ABANICO: Distribuimos los 3 puntos positivos a la derecha equitativamente
    if mid_pos:
        y_coords_pos = np.linspace(mid_pos[0]['Pn'] + (max_P - min_P)*0.05, mid_pos[-1]['Pn'] - (max_P - min_P)*0.05, len(mid_pos))
        for i, pt in enumerate(mid_pos):
            anotar_punto(pt, max_M + (max_M - min_M)*0.25, y_coords_pos[i], 'left', 'center')
            
    # 📌 ESTRATEGIA DE ABANICO: Distribuimos los 3 puntos negativos a la izquierda equitativamente
    if mid_neg:
        y_coords_neg = np.linspace(mid_neg[0]['Pn'] + (max_P - min_P)*0.05, mid_neg[-1]['Pn'] - (max_P - min_P)*0.05, len(mid_neg))
        for i, pt in enumerate(mid_neg):
            anotar_punto(pt, min_M - (max_M - min_M)*0.25, y_coords_neg[i], 'right', 'center')
            
    ax.axhline(0, color='black', lw=1); ax.axvline(0, color='black', lw=1)
    ax.set_xlabel("Momento Flexionante Mn / Mu (ton-m)", fontweight='bold', fontsize=11)
    ax.set_ylabel("Carga Axial Pn / Pu (ton)", fontweight='bold', fontsize=11)
    ax.set_title(titulo, fontweight='bold', fontsize=14)
    ax.grid(True, linestyle=':', alpha=0.5)
    
    # Expandir los límites del gráfico para que quepan las cajas de texto laterales
    rango_M = max_M - min_M
    rango_P = max_P - min_P
    ax.set_xlim(min_M - rango_M * 0.45, max_M + rango_M * 0.45)
    ax.set_ylim(min_P - rango_P * 0.20, max_P + rango_P * 0.20)
    
    ax.legend(loc='lower left', frameon=True, fontsize=10)
    return fig

def mostrar_tabla_estilizada(pts):
    df = generar_tabla_resumen(pts)
    st.dataframe(df.style.format({
        "Mn (ton-m)": "{:.2f}", "Pn (ton)": "{:.2f}", 
        "Factor FR": "{:.3f}", "Mu (ton-m)": "{:.2f}", "Pu (ton)": "{:.2f}"
    }).set_properties(**{'background-color': '#f8f9fa', 'text-align': 'center'}), use_container_width=True, hide_index=True)

def highlight_zero_force(row):
    """ Resalta en verde la fila de la varilla cuya fuerza se anula (fs=0) """
    try:
        val = str(row["Fuerza (kg)"]).replace(',', '')
        if abs(float(val)) < 0.01:
            return ['background-color: #d4edda; color: #155724; font-weight: bold'] * len(row)
    except:
        pass
    return [''] * len(row)

def imprimir_memoria_punto(res):
    st.markdown(f"#### 🔹 Punto [{res['id']}]: {res['nombre'].replace('(+)', '').replace('(-)', '')}")
    
    if res['c'] == float('inf'):
        st.markdown(f"**Eje Neutro (c):** $\infty$ (Compresión Pura) | **Bloque de Compresión (a):** {res['a']:.3f} cm")
        st.markdown(f"**Fórmula Teórica:** $P_o = 0.85 f'_c (A_g - A_s) + \sum A_s f_y$")
    elif res['c'] == 0.0:
        st.markdown(f"**Eje Neutro (c):** 0.00 cm (Tensión Pura) | **Bloque de Compresión (a):** 0.00 cm")
        st.markdown(f"**Fórmula Teórica:** $P_n = \sum A_s (-f_y)$. *El concreto no resiste tensión.*")
    else:
        st.markdown(f"**Eje Neutro (c):** {res['c']:.3f} cm | **Bloque de Compresión (a):** {res['a']:.3f} cm")
        if 'Conveniente' in res['nombre']:
            st.caption("💡 *Nota: Se eligió estratégicamente este valor de 'c' porque coincide con la profundidad de un lecho de acero, anulando su deformación y simplificando el cálculo.*")

    col1, col2 = st.columns(2)
    with col1:
        st.write("**1. Fuerzas en el Concreto**")
        if res['det_c']:
            df_c = pd.DataFrame(res['det_c'])
            df_c.loc['TOTAL'] = df_c.sum(numeric_only=True)
            df_c.at['TOTAL', 'Elemento'] = "SUMATORIA"
            st.dataframe(df_c.style.format(precision=2), hide_index=True)
        else:
            st.info("El concreto no aporta resistencia en este estado de análisis.")

    with col2:
        st.write("**2. Fuerzas en el Acero**")
        if res['det_s']:
            df_s = pd.DataFrame(res['det_s'])
            df_s.loc['TOTAL'] = df_s.sum(numeric_only=True)
            df_s.at['TOTAL', 'Elemento'] = "SUMATORIA"
            
            st.dataframe(df_s.style.format({
                "Def unit. (ε)": lambda x: f"{x}" if isinstance(x, str) else f"{x:.5f}", 
                "fs neto (kg/cm²)": "{:.2f}", "Fuerza (kg)": "{:.2f}", 
                "Brazo a Yp (cm)": "{:.3f}", "Momento (kg-cm)": "{:.2f}"
            }).apply(highlight_zero_force, axis=1), hide_index=True)
            
    st.success(f"**NOMINAL:** Pn = {res['Pn']:,.2f} ton, Mn = {res['Mn']:,.2f} ton-m | **FR = {res['FR']:.3f}** | **DISEÑO:** Pu = {res['Pu']:,.2f} ton, Mu = {res['Mu']:,.2f} ton-m")
    st.markdown("---")

def dibujar_seccion(sec, titulo):
    fig, ax = plt.subplots(figsize=(4, 5))
    max_b = max([b.b for b in sec.bloques]) if sec.bloques else 40
    
    for b in sec.bloques:
        rect = patches.Rectangle((-b.b/2, b.y_top), b.b, b.h, linewidth=2.5, edgecolor='#2c3e50', facecolor='#ecf0f1', hatch='//')
        ax.add_patch(rect)
        
    for l in sec.lechos:
        w = max_b
        for b in sec.bloques:
            if b.y_top <= l.d <= b.y_bot: 
                w = b.b; break
        xs = np.linspace(-w/2 + 4, w/2 - 4, int(l.cant)) if l.cant > 1 else [0]
        for x in xs:
            ax.plot(x, l.d, 'o', color='#8e44ad', markersize=10, markeredgecolor='black', markeredgewidth=1.5)
            
    ax.axhline(sec.Yp, color='#e74c3c', linestyle='--', linewidth=3, label=f'Yp={sec.Yp:.2f}cm')
    ax.set_aspect('equal', 'box')
    ax.set_xlim(-max_b/2 - 5, max_b/2 + 5)
    ax.set_ylim(sec.H + 5, -5) 
    ax.axis('off')
    ax.set_title(titulo, fontweight='bold', fontsize=13)
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    return fig

def imprimir_centroide(sec):
    st.markdown(f"### 📍 Centroide Plástico ($Y_p$)")
    st.caption("Cálculo del centro geométrico balanceado respecto a la fibra de compresión extrema.")
    df_yp = pd.DataFrame(sec.detalles_yp)
    df_yp.loc['Total'] = df_yp.sum(numeric_only=True)
    df_yp.at['Total', 'Elemento'] = "SUMATORIA"
    st.dataframe(df_yp.style.format(precision=2), hide_index=True)
    st.info(f"**$P_o$ = {sec.Po/1000:,.2f} ton** | **$Y_p$ = {sec.Yp:.3f} cm** (medidos desde la fibra superior)")

# ============================================================================
# APLICACIÓN STREAMLIT
# ============================================================================
def main():
    st.set_page_config(layout="wide", page_title="Cálculo Columnas NTC")
    st.title("🏛️ Diseño de Columnas NTC 2023")
    st.markdown("Cálculo Teórico Riguroso: Ceros de Tensión en Concreto, Punto Conveniente Inteligente (c=d), Tablas Generales y Gráficas Impecables.")
    
    modo = st.tabs(["1. Columna Rectangular (Ejes Separados X/Y)", "2. Sección Asimétrica (Bloques SAP)"])

    # -------------------------------------------------------------------------
    # MÓDULO 1: COLUMNA RECTANGULAR MONOLÍTICA
    # -------------------------------------------------------------------------
    with modo[0]:
        st.info("Configura los aceros para el Eje Fuerte (cortando `h`) y para el Eje Débil (cortando `b`) de manera independiente.")
        colA, colB, colC, colD = st.columns(4)
        b = colA.number_input("Base b (cm)", value=30.0, step=5.0)
        h = colB.number_input("Peralte h (cm)", value=40.0, step=5.0)
        fc = colC.number_input("f'c (kg/cm²)", value=300.0, step=50.0)
        fy = colD.number_input("fy (kg/cm²)", value=4200.0, step=100.0)
        
        st.markdown("### ⚙️ Disposición de Aceros por Ejes")
        cx, cy = st.columns(2)
        
        with cx:
            st.markdown("#### Aceros EJE FUERTE (Flexión sobre h)")
            st.caption("Distancia `d` medida desde la fibra de compresión extrema.")
            n_lechos_x = st.number_input("Número de Lechos (Eje Fuerte)", 2, 15, 3, key='nx')
            lechos_fuerte = []
            for i in range(int(n_lechos_x)):
                c_a, c_b, c_c = st.columns(3)
                default_dx = 5.0 if i==0 else (h-5.0 if i==n_lechos_x-1 else h/2)
                d = c_a.number_input(f"d L{i+1} (cm)", value=float(default_dx), step=1.0, key=f'dx_{i}')
                var = c_b.number_input(f"Varilla #", value=6.0, step=1.0, key=f'vx_{i}')
                cant = c_c.number_input(f"Cant.", value=3, step=1, key=f'cx_{i}')
                As = cant * AREAS_VARILLAS.get(var, (np.pi*(var*2.54/8)**2)/4)
                lechos_fuerte.append(Lecho(As, d, cant))
                
        with cy:
            st.markdown("#### Aceros EJE DÉBIL (Flexión sobre b)")
            st.caption("Distancia `d` medida desde la fibra de compresión extrema.")
            n_lechos_y = st.number_input("Número de Lechos (Eje Débil)", 2, 15, 3, key='ny')
            lechos_debil = []
            for i in range(int(n_lechos_y)):
                c_a, c_b, c_c = st.columns(3)
                default_dy = 5.0 if i==0 else (b-5.0 if i==n_lechos_y-1 else b/2)
                d = c_a.number_input(f"d L{i+1} (cm)", value=float(default_dy), step=1.0, key=f'dy_{i}')
                var = c_b.number_input(f"Varilla #", value=6.0, step=1.0, key=f'vy_{i}')
                cant = c_c.number_input(f"Cant.", value=2, step=1, key=f'cy_{i}')
                As = cant * AREAS_VARILLAS.get(var, (np.pi*(var*2.54/8)**2)/4)
                lechos_debil.append(Lecho(As, d, cant))

        if st.button("🚀 Calcular Ambos Ejes (Fuerte y Débil)", type='primary', use_container_width=True):
            
            sec_fuerte = SeccionColumna(fc, fy, [Bloque(b, h, 0)], lechos_fuerte)
            sec_debil = SeccionColumna(fc, fy, [Bloque(h, b, 0)], lechos_debil)
            
            tf, td = st.tabs(["📊 RESULTADOS EJE FUERTE", "📊 RESULTADOS EJE DÉBIL"])
            
            with tf:
                st.markdown(f"### EJE FUERTE (Flexión sobre h={h} cm)")
                pts_fuerte = sec_fuerte.obtener_lista_puntos_clave()
                
                st.markdown("#### 📋 TABLA GENERAL DE RESULTADOS")
                mostrar_tabla_estilizada(pts_fuerte)
                
                col1, col2 = st.columns([1, 2.5])
                with col1:
                    st.pyplot(dibujar_seccion(sec_fuerte, f"Vista Eje Fuerte\n(Yp = {sec_fuerte.Yp:.2f} cm)"))
                with col2:
                    st.pyplot(plot_diagrama_con_flechas(sec_fuerte, "Diagrama Biaxial Numerado (Eje Fuerte)"))
                
                st.markdown("---")
                imprimir_centroide(sec_fuerte)
                st.markdown("### 🧮 MEMORIA DE CÁLCULO DESGLOSADA (EJE FUERTE)")
                for pt in pts_fuerte:
                    imprimir_memoria_punto(pt)

            with td:
                st.markdown(f"### EJE DÉBIL (Flexión sobre b={b} cm)")
                pts_debil = sec_debil.obtener_lista_puntos_clave()
                
                st.markdown("#### 📋 TABLA GENERAL DE RESULTADOS")
                mostrar_tabla_estilizada(pts_debil)
                
                col1, col2 = st.columns([1, 2.5])
                with col1:
                    st.pyplot(dibujar_seccion(sec_debil, f"Vista Eje Débil\n(Yp = {sec_debil.Yp:.2f} cm)"))
                with col2:
                    st.pyplot(plot_diagrama_con_flechas(sec_debil, "Diagrama Biaxial Numerado (Eje Débil)"))
                
                st.markdown("---")
                imprimir_centroide(sec_debil)
                st.markdown("### 🧮 MEMORIA DE CÁLCULO DESGLOSADA (EJE DÉBIL)")
                for pt in pts_debil:
                    imprimir_memoria_punto(pt)

    # -------------------------------------------------------------------------
    # MÓDULO 2: SECCIÓN ASIMÉTRICA POR BLOQUES (TIPO SAP)
    # -------------------------------------------------------------------------
    with modo[1]:
        st.info("💡 **El Punto Conveniente** detecta automáticamente los lechos intermedios para anular sus fuerzas al asignar `C = d_i`")
        colP, colC = st.columns([1, 2.5])
        with colP:
            st.markdown("### Parámetros Generales")
            fc_s = st.number_input("f'c Concreto (kg/cm²)", value=350.0, step=10.0, key='fcs')
            fy_s = st.number_input("fy Acero (kg/cm²)", value=4200.0, step=100.0, key='fys')
            
            st.markdown("#### 🧱 Bloques de Concreto")
            df_b = pd.DataFrame([{"b (cm)":40.0, "h (cm)":20.0, "Y_top (cm)":0.0}, {"b (cm)":20.0, "h (cm)":20.0, "Y_top (cm)":20.0}])
            b_edit = st.data_editor(df_b, num_rows="dynamic", hide_index=True)
            
            st.markdown("#### ⚙️ Lechos de Acero")
            df_l = pd.DataFrame([{"d (cm)":4.0, "As (cm²)":11.4, "Cant":4, "Var#":6.0}, {"d (cm)":16.0, "As (cm²)":5.7, "Cant":2, "Var#":6.0}, {"d (cm)":36.0, "As (cm²)":5.7, "Cant":2, "Var#":6.0}])
            l_edit = st.data_editor(df_l, num_rows="dynamic", hide_index=True)
            
            btn_sap = st.button("🚀 Calcular Sección Asimétrica", type='primary', use_container_width=True)

        with colC:
            if btn_sap:
                b_list = [Bloque(r["b (cm)"], r["h (cm)"], r["Y_top (cm)"]) for _, r in b_edit.iterrows() if r["b (cm)"]>0]
                l_list = [Lecho(r["As (cm²)"] if r["As (cm²)"]>0 else r["Cant"]*AREAS_VARILLAS.get(r["Var#"], 0), r["d (cm)"], r["Cant"]) for _, r in l_edit.iterrows()]
                
                sec_sap = SeccionColumna(fc_s, fy_s, b_list, l_list)
                pts_sap = sec_sap.obtener_lista_puntos_clave()
                
                st.markdown("### 📋 TABLA GENERAL DE RESULTADOS (SAP)")
                mostrar_tabla_estilizada(pts_sap)
                
                tv, tg, tm = st.tabs(["📈 Diagrama y Vista", "📖 Memoria (Paso a Paso)", "📍 Centroide Plástico"])
                
                with tv:
                    col_v1, col_v2 = st.columns([1, 2])
                    with col_v1:
                        st.pyplot(dibujar_seccion(sec_sap, f"Sección\n(Yp = {sec_sap.Yp:.2f} cm)"))
                    with col_v2:
                        st.pyplot(plot_diagrama_con_flechas(sec_sap, "Diagrama Biaxial (Asimétrica)"))
                    
                with tm:
                    for pt in pts_sap: 
                        imprimir_memoria_punto(pt)
                        
                with tm: # Reusamos para imprimir el centroide
                    pass
                    
                with tm:
                    pass

                with tm:
                    imprimir_centroide(sec_sap)

if __name__ == "__main__":
    import sys, subprocess
    if st.runtime.exists():
        main()
    else:
        subprocess.run([sys.executable, "-m", "streamlit", "run", sys.argv[0]])