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
        value='3x + 2y <= 18\nx + y <= 5\nx <= 4\nx >= 0\ny >= 0',
        style={'width': '300px', 'height': '120px'}
    ),

    html.Br(),
    html.Button("Resolver", id='btn'),
    html.Br(), html.Br(),

    html.Img(id='grafico', style={'maxWidth': '600px'})
])


def parse_expr(expr):
    expr = expr.replace(' ', '')
    a, b = 0, 0

    # Garantir que "x" e "y" sozinhos sejam reconhecidos corretamente
    x_match = re.search(r'([+-]?\d*\.?\d*)x', expr)
    y_match = re.search(r'([+-]?\d*\.?\d*)y', expr)

    if x_match:
        x_val = x_match.group(1)
        a = float(x_val) if x_val not in ['', '+', '-'] else float(x_val + '1')

    if y_match:
        y_val = y_match.group(1)
        b = float(y_val) if y_val not in ['', '+', '-'] else float(y_val + '1')

    return a, b


@app.callback(
    Output('grafico', 'src'),
    Input('btn', 'n_clicks'),
    State('objetivo', 'value'),
    State('tipo', 'value'),
    State('restricoes', 'value')
)
def resolver(n, objetivo, tipo, restricoes):
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

        # Inicia o gráfico e cria uma grade para preencher a região viável
        fig, ax = plt.subplots()
        x = np.linspace(0, 10, 400)
        y = np.linspace(0, 10, 400)
        X, Y = np.meshgrid(x, y)
        mask = np.ones_like(X, dtype=bool)

        tol = 1e-9
        # Para cada restrição, atualiza a máscara e plota a reta correspondente
        for i in range(len(A)):
            a1, a2 = A[i]
            if abs(a1) < tol and abs(a2) < tol:
                print(f"Aviso: Restrição {i+1} inválida (coeficientes zero), ignorada.")
                continue
            # Atualiza a máscara para a região onde a restrição é satisfeita
            mask &= (a1 * X + a2 * Y <= b[i])
            # Plota a restrição
            if abs(a2) > tol:
                y_vals = (b[i] - a1 * x) / a2
                ax.plot(x, y_vals, label=f'Restrição {i + 1}')
            elif abs(a1) > tol:
                x_const = b[i] / a1
                ax.axvline(x_const, label=f'Restrição {i + 1}')
            else:
                print(f"Aviso: Restrição {i+1} inválida (coeficientes zero), ignorada.")

        # Preenche a região viável com uma cor (neste exemplo, #c0ffee)
        ax.contourf(X, Y, mask, levels=[0.5, 1], colors=['#c0ffee'], alpha=0.5)

        # Plota o ponto ótimo, se houver solução
        if res.success:
            x_opt, y_opt = res.x
            ax.plot(x_opt, y_opt, 'ro', label='Ponto ótimo')
            ax.text(x_opt, y_opt, f'({x_opt:.2f}, {y_opt:.2f})', fontsize=10)
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
