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

    html.Label("Função objetivo (ex: 4x + 8y):"),
    dcc.Input(id='objetivo', type='text', value='4x + 8y', style={'width': '300px'}),

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

    html.Label("Restrições (uma por linha):"),
    dcc.Textarea(
        id='restricoes',
        value='8x + 4y <= 1280\n4x + 12y <= 1600\nx >= 100\ny >= 100\n4x + 4y <= 760\nx <= 140\ny <= 120',
        style={'width': '300px', 'height': '140px'}
    ),

    html.Br(),
    html.Button("Resolver", id='btn'),
    html.Br(), html.Br(),

    html.Img(id='grafico', style={'maxWidth': '700px'})
])


def parse_expr(expr):
    expr = expr.replace(' ', '')
    a, b = 0, 0
    x_match = re.search(r'([+-]?\d*\.?\d*)x', expr)
    y_match = re.search(r'([+-]?\d*\.?\d*)y', expr)
    if x_match:
        v = x_match.group(1)
        a = float(v) if v not in ['', '+', '-'] else float(v + '1')
    if y_match:
        v = y_match.group(1)
        b = float(v) if v not in ['', '+', '-'] else float(v + '1')
    return a, b


@app.callback(
    Output('grafico', 'src'),
    Input('btn', 'n_clicks'),
    State('objetivo', 'value'),
    State('tipo', 'value'),
    State('restricoes', 'value'),
)
def resolver(n, objetivo, tipo, restricoes):
    if n is None:
        return dash.no_update

    try:
        # parse objetivo
        c = np.array(parse_expr(objetivo))
        if tipo == 'max':
            c = -c

        # parse restrições
        A, b = [], []
        for linha in restricoes.strip().split('\n'):
            if '<=' in linha:
                lhs, rhs = linha.split('<=')
                coeff = parse_expr(lhs)
                A.append(coeff)
                b.append(float(rhs))
            elif '>=' in linha:
                lhs, rhs = linha.split('>=')
                coeff = parse_expr(lhs)
                coeff = [-coeff[0], -coeff[1]]
                A.append(coeff)
                b.append(-float(rhs))
            elif '=' in linha:
                lhs, rhs = linha.split('=')
                coeff = parse_expr(lhs)
                A.append(coeff)
                b.append(float(rhs))

        # resolve PL
        res = linprog(c, A_ub=A, b_ub=b, bounds=(0, None), method='highs')

        # coletar pontos de intercepto para definir eixos
        pts = [(0, 0)]
        tol = 1e-9
        for ai, bi in zip(A, b):
            a1, a2 = ai
            # x-intercept: y=0 => x = bi/a1
            if abs(a1) > tol:
                pts.append((bi/a1, 0))
            # y-intercept: x=0 => y = bi/a2
            if abs(a2) > tol:
                pts.append((0, bi/a2))
        # incluir ponto ótimo
        if res.success:
            pts.append(tuple(res.x))

        xs, ys = zip(*pts)
        xmin, xmax = min(xs), max(xs)
        ymin, ymax = min(ys), max(ys)
        # margem de 10%
        mx = (xmax - xmin) * 0.1 if xmax > xmin else xmax * 0.1
        my = (ymax - ymin) * 0.1 if ymax > ymin else ymax * 0.1
        xmin, xmax = max(0, xmin - mx), xmax + mx
        ymin, ymax = max(0, ymin - my), ymax + my

        # preparar grid e máscara
        x = np.linspace(xmin, xmax, 400)
        y = np.linspace(ymin, ymax, 400)
        X, Y = np.meshgrid(x, y)
        mask = np.ones_like(X, dtype=bool)

        # plot
        fig, ax = plt.subplots()
        for idx, (ai, bi) in enumerate(zip(A, b), start=1):
            a1, a2 = ai
            mask &= (a1*X + a2*Y <= bi)
            # reta
            if abs(a2) > tol:
                Yc = (bi - a1*x) / a2
                ax.plot(x, Yc, label=f'Restrição {idx}')
            elif abs(a1) > tol:
                xc = bi/a1
                ax.axvline(x=xc, label=f'Restrição {idx}')

        # região viável
        ax.contourf(X, Y, mask, levels=[0.5, 1], colors=['#c0ffee'], alpha=0.5)

        # ponto ótimo
        if res.success:
            x_opt, y_opt = res.x
            ax.plot(x_opt, y_opt, 'ro', label='Ponto ótimo')
            ax.text(x_opt, y_opt, f'({x_opt:.2f}, {y_opt:.2f})')
        else:
            ax.text((xmin+xmax)/2, (ymin+ymax)/2,
                    'Sem solução viável', color='red', ha='center')

        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.grid(True)
        ax.legend()

        # converter para base64
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        plt.close(fig)
        buf.seek(0)
        img = base64.b64encode(buf.read()).decode('utf-8')
        return f"data:image/png;base64,{img}"

    except Exception as e:
        print("Erro no callback:", e)
        return dash.no_update


if __name__ == '__main__':
    app.run(debug=True)
