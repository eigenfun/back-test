from nicegui import ui
import subprocess
import pandas as pd
from io import StringIO
import re
import plotly.graph_objects as go
import asyncio

async def run_backtest():
    """Run the selected backtesting script."""
    run_button.disable()
    raw_output.set_content("Running backtest...")

    command = [
        'python',
        strategy.value,
        symbol.value,
        '--start-date', start_date.value,
        '--end-date', end_date.value,
        '--capital', str(capital.value),
        '--deltas', deltas.value,
        '--weekly'  # Always request weekly data for parsing
    ]

    try:
        # Run the script in a separate process
        process = await asyncio.create_subprocess_exec(
            *command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        stdout, stderr = await process.communicate()

        if process.returncode == 0:
            output = stdout.decode()
            raw_output.set_content(output)
            parse_and_display_results(output)
        else:
            raw_output.set_content(f"Error running script:\n{stderr.decode()}")

    except FileNotFoundError:
        raw_output.set_content(f"Error: '{strategy.value}' not found. Make sure it's in the correct path.")
    finally:
        run_button.enable()


def parse_and_display_results(output: str):
    """Parse the raw output and display it in the UI."""
    # Parse summary
    summary_match = re.search(r"STRATEGY PERFORMANCE SUMMARY\n=+\n(.*?)\n=+", output, re.DOTALL)
    if summary_match:
        summary_output.set_content(summary_match.group(1).strip())
    else:
        summary_output.set_content("Could not parse summary.")

    # Parse weekly data
    weekly_match = re.search(r"WEEKLY POSITION AND P\&L SUMMARY.*?\n=+\n(.*?)\n=+", output, re.DOTALL)
    if weekly_match:
        weekly_data_str = weekly_match.group(1)
        # Use pandas to read the fixed-width format
        try:
            # A bit of cleaning to handle the header separator
            lines = weekly_data_str.strip().split('\n')
            header = lines[0]
            separator = lines[1]
            data_lines = lines[2:]

            # Reconstruct the string for pandas
            data_io = StringIO('\n'.join([header] + data_lines))

            # Use regex to split columns, as spaces can be inconsistent
            df = pd.read_fwf(data_io)

            # Display in an AG Grid
            weekly_grid.options['rowData'] = df.to_dict('records')
            weekly_grid.update()

            # Create chart
            df['date'] = pd.to_datetime(df['Date'])
            df['total_value'] = df['Total Value'].replace({r'\$': '', ',': ''}, regex=True).astype(float)

            fig = go.Figure()
            fig.add_trace(go.Scatter(x=df['date'], y=df['total_value'], mode='lines', name='Total Value'))
            fig.update_layout(title='Portfolio Value Over Time', xaxis_title='Date', yaxis_title='Total Value ($)')
            chart.figure = fig
            chart.update()

        except Exception as e:
            weekly_grid.options['rowData'] = []
            weekly_grid.update()
            raw_output.set_content(f"Error parsing weekly data: {e}\n\n{weekly_data_str}")

    else:
        weekly_grid.options['rowData'] = []
        weekly_grid.update()


# UI Layout
with ui.header():
    ui.label('Options Backtesting')

with ui.row():
    with ui.card():
        ui.label('Backtest Configuration').classes('text-h6')

        strategy = ui.select(
            ['cash_secured_puts.py', 'covered_calls.py', 'hybrid.py'],
            value='cash_secured_puts.py',
            label='Strategy'
        )

        symbol = ui.input('Symbol', value='NVDA')
        start_date = ui.input('Start Date', value='2024-01-01')
        end_date = ui.input('End Date', value='2024-02-01')
        capital = ui.number('Initial Capital', value=100000, format='%.0f')
        deltas = ui.input('Target Delta(s)', value='0.3')

        run_button = ui.button('Run Backtest', on_click=run_backtest)

with ui.row():
    with ui.card().classes('w-full'):
        ui.label('Results').classes('text-h6')

        with ui.tabs() as tabs:
            ui.tab('Summary')
            ui.tab('Weekly Data')
            ui.tab('Chart')
            ui.tab('Raw Output')

        with ui.tab_panels(tabs, value='Summary'):
            with ui.tab_panel('Summary'):
                summary_output = ui.code('Summary will be shown here.')
            with ui.tab_panel('Weekly Data'):
                weekly_grid = ui.aggrid({
                    'defaultColDef': {'resizable': True, 'sortable': True},
                    'columnDefs': [],
                    'rowData': [],
                })
            with ui.tab_panel('Chart'):
                chart = ui.plotly(go.Figure())
            with ui.tab_panel('Raw Output'):
                raw_output = ui.code('Raw output will be shown here.').classes('w-full h-64')

ui.run()
