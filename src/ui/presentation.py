"""Styling only: never fit models, change data, or recalculate metrics."""
import streamlit as st
import textwrap
from src.utils.paths import ROOT


def load_custom_css():
    css = (ROOT / 'assets/styles.css').read_text(encoding='utf-8')
    st.markdown(f'<style>{css}</style>', unsafe_allow_html=True)


def style_chart(figure):
    if figure.layout.title.text:
        figure.update_layout(title_text='<br>'.join(textwrap.wrap(figure.layout.title.text.replace('<br>', ' '), width=28)))
    figure.update_layout(
        template='plotly_white', paper_bgcolor='#FFFFFF', plot_bgcolor='#FFFFFF',
        font=dict(family='Arial, sans-serif', size=14, color='#172033'),
        title=dict(font=dict(size=18, color='#172033'), x=.02),
        colorway=['#152238', '#FF5252', '#166534', '#64748B', '#E63E45'],
        autosize=True, height=480, margin=dict(l=30, r=25, t=80, b=60),
        legend=dict(orientation='h', y=-.25, x=0, title_text=''),
        hoverlabel=dict(bgcolor='#FFFFFF', font_color='#172033'))
    figure.update_xaxes(automargin=True, gridcolor='#DCE3ED', zerolinecolor='#64748B')
    figure.update_yaxes(automargin=True, gridcolor='#DCE3ED', zerolinecolor='#64748B')
    if figure.layout.yaxis.title.text == 'count':
        figure.update_layout(yaxis_title='Số căn nhà')
    # Plotly Express writes explicit marker colors, which override layout.colorway.
    palette = ['#152238', '#FF5252', '#166534', '#64748B', '#E63E45']
    groups = {}
    for trace in figure.data:
        if trace.type in ('scatter', 'bar', 'histogram', 'box'):
            color = trace.marker.color
            if color is None or isinstance(color, str):
                key = trace.name or 'default'
                groups.setdefault(key, palette[len(groups) % len(palette)])
                trace.update(marker_color=groups[key])
                if trace.type in ('scatter', 'box'):
                    trace.update(line_color=groups[key])
    # px.imshow correlation uses a diverging scale; keep its quantitative encoding.
    has_numeric_color = any(trace.type == 'scatter' and trace.marker.color is not None
                            and not isinstance(trace.marker.color, str) for trace in figure.data)
    is_heatmap = any(trace.type == 'heatmap' for trace in figure.data)
    if has_numeric_color:
        figure.update_layout(coloraxis_colorscale=[[0, '#64748B'], [1, '#152238']])
    if has_numeric_color or is_heatmap:
        # A horizontal scale leaves room for the plot at phone widths.
        figure.update_layout(coloraxis_colorbar=dict(orientation='h', x=.5, xanchor='center',
            y=-.35, len=.85, thickness=12, title=dict(side='top')), margin_b=120)
    if is_heatmap:
        labels = ['DT sống', 'Phòng ngủ', 'Phòng tắm', 'Garage', 'Năm XD', 'Chất lượng', 'DT hầm', 'Giá']
        trace = next(t for t in figure.data if t.type == 'heatmap')
        figure.update_xaxes(tickmode='array', tickvals=list(trace.x), ticktext=labels, tickangle=-45, tickfont_size=10)
        figure.update_yaxes(tickmode='array', tickvals=list(trace.y), ticktext=labels, tickfont_size=10)
    return figure


def show_chart(figure):
    st.plotly_chart(style_chart(figure), width='stretch', theme=None,
                    config={'responsive': True, 'displaylogo': False})
    if any(trace.type == 'heatmap' for trace in figure.data):
        st.caption('DT sống: GrLivArea · DT hầm: TotalBsmtSF · Năm XD: YearBuilt. Di chuột hoặc mở toàn màn hình để xem hệ số tương quan chi tiết.')
