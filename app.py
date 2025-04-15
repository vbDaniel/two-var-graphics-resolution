import matplotlib
matplotlib.use('Agg')  # Força o uso do backend 'Agg' sem interface gráfica

import dash
from dash import html, dcc, Input, Output, State
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import linprog
import io
import base64
import re

app = dash.Dash(__name__)
server = app.server

app.layout = html.Div([
    html.H1("Solução Gráfica de Programação Linear"),

    html.Label("Função objetivo (ex: 3x + 2y):"),
    dcc.Input(id='objetivo', type='text', value='3x + 2y', style={'width': '300px'}),

    html.Label("Tipo (max ou min):"),
    dcc.Dropdown(
        id='tipo', 
        options=[
            {'label': 'Maximizar', 'value': 'max'},
            {'label': 'Minimizar', 'value': 'min'}
        ],
        value='max', 
        style={'width': '150px'}
    ),

    html.Label("Restrições (uma por linha, ex: x + y <= 4):"),
    dcc.Textarea(
        id='restricoes', 
        value='x + y <= 4\nx >= 0\ny >= 0',
        style={'width': '300px', 'height': '120px'}
    ),

    html.Br(),
    html.Button("Resolver", id='btn'),
    html.Br(), html.Br(),

    html.Img(id='grafico', style={'maxWidth': '600px'})
])


def parse_expr(expr):
    """
    Parseia uma expressão do tipo '3x+2y' e retorna uma tupla com os coeficientes (coef_x, coef_y).
    """
    expr = expr.replace(' ', '')
    match = re.findall(r'([+-]?\d*\.?\d*)x|([+-]?\d*\.?\d*)y', expr)
    a, b = 0, 0
    for x, y in match:
        if x:
            a = float(x) if x not in ['', '+', '-'] else float(x + '1')
        if y:
            b = float(y) if y not in ['', '+', '-'] else float(y + '1')
    return a, b


@app.callback(
    Output('grafico', 'src'),
    Input('btn', 'n_clicks'),
    State('objetivo', 'value'),
    State('tipo', 'value'),
    State('restricoes', 'value')
)
def resolver(n, objetivo, tipo, restricoes):
    # Verifica se o botão foi clicado
    if n is None:
        return dash.no_update
    try:
        # Processa a função objetivo
        c = np.array(parse_expr(objetivo))
        if tipo == 'max':
            c = -c  # O linprog minimiza por padrão

        A = []
        b = []
        linhas = restricoes.strip().split('\n')
        # Processa cada restrição
        for linha in linhas:
            if '<=' in linha:
                partes = linha.split('<=')
                a_i = parse_expr(partes[0])
                b_i = float(partes[1])
                A.append(a_i)
                b.append(b_i)
            elif '>=' in linha:
                partes = linha.split('>=')
                a_i = parse_expr(partes[0])
                a_i = [-i for i in a_i]  # Transforma o sinal para <=
                b_i = -float(partes[1])
                A.append(a_i)
                b.append(b_i)
            elif '=' in linha:
                partes = linha.split('=')
                a_i = parse_expr(partes[0])
                b_i = float(partes[1])
                A.append(a_i)
                b.append(b_i)

        res = linprog(c, A_ub=A, b_ub=b, bounds=(0, None), method='highs')

        fig, ax = plt.subplots()
        x_vals = np.linspace(0, 10, 400)
        for i in range(len(A)):
            a1, a2 = A[i]
            # Defina uma tolerância pequena para evitar divisões por zero
            tol = 1e-9
            if abs(a2) > tol:
                y = (b[i] - a1 * x_vals) / a2
                ax.plot(x_vals, y, label=f'Restrição {i + 1}')
            elif abs(a1) > tol:  # Se a2 é zero, mas a1 não é, plota uma linha vertical
                x_const = b[i] / a1
                ax.axvline(x_const, label=f'Restrição {i + 1}')
            else:
                # Se ambos forem praticamente zero, não há uma reta definida.
                print(f"Aviso: Restrição {i+1} inválida (coeficientes zero), ignorada.")

        if res.success:
            x, y = res.x
            ax.plot(x, y, 'ro', label='Ponto ótimo')
            ax.text(x, y, f'({x:.2f}, {y:.2f})', fontsize=10)
        else:
            ax.text(2, 2, 'Problema sem solução viável!', color='red')

        ax.set_xlim(0, 10)
        ax.set_ylim(0, 10)
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.grid(True)
        ax.legend()

        buf = io.BytesIO()
        plt.savefig(buf, format="png")
        plt.close(fig)
        buf.seek(0)
        img_b64 = base64.b64encode(buf.read()).decode('utf-8')
        return f"data:image/png;base64,{img_b64}"
    except Exception as e:
        print("Erro no callback:", e)
        return dash.no_update


if __name__ == '__main__':
    app.run(debug=True)
