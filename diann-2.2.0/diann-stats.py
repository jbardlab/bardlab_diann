import polars as pl
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

HIST_COLOR = 'steelblue'
CMAP = 'bone_r'

# Plot configurations: toggle 'enabled' to True/False to show/hide plots
plot_configs = {
    'Identified_MS1_TIC': {'function': 'plot_identified_ms1_tic', 'enabled': True},
    'RT_histogram': {'function': 'plot_rt_histogram', 'enabled': True},
    'MZ_histogram': {'function': 'plot_mz_histogram', 'enabled': True},
    'Charge_histogram': {'function': 'plot_charge_histogram', 'enabled': True},
    'Precursors_per_protein': {'function': 'plot_precursors_per_protein', 'enabled': True},
    'MS1_corr_histogram': {'function': 'plot_ms1_corr_histogram', 'enabled': True},
    'RT_vs_iRT': {'function': 'plot_rt_vs_irt', 'enabled': True},
    'RT_vs_Predicted_RT': {'function': 'plot_rt_vs_predicted_rt', 'enabled': True},
    'MZ_vs_RT': {'function': 'plot_mz_vs_rt', 'enabled': True},
    'IM_vs_Predicted_IM': {'function': 'plot_im_vs_predicted_im', 'enabled': True},
    'IM_vs_MZ': {'function': 'plot_im_vs_mz', 'enabled': True},
    'Normalization_vs_RT': {'function': 'plot_normalization_vs_rt', 'enabled': True},
    'MZ_delta_vs_RT': {'function': 'plot_mz_delta_vs_rt', 'enabled': True},
    'MZ_delta_vs_MZ': {'function': 'plot_mz_delta_vs_mz', 'enabled': True},
    'MS2_delta_vs_RT': {'function': 'plot_ms2_delta_vs_rt', 'enabled': True},
    'MS2_delta_vs_MZ': {'function': 'plot_ms2_delta_vs_mz', 'enabled': True},
    'FWHM_histogram': {'function': 'plot_fwhm_histogram', 'enabled': True},
    'FWHM_vs_RT': {'function': 'plot_fwhm_vs_rt', 'enabled': True},
    'QValue_histogram': {'function': 'plot_qvalue_histogram', 'enabled': False},
    'IRT_proportion': {'function': 'plot_irt_proportion', 'enabled': True},
    'MZ_proportion_global': {'function': 'plot_mz_proportion_global', 'enabled': True},
}

run_to_proportions = None
irt_bin_centers = None
mz_bin_centers = None

def plot_rt_histogram(df, run, ax):
    try:
        ax.hist(df['RT'], bins=50, color=HIST_COLOR, rwidth=0.9)
        ax.set_title(f"RT, n = {df.height}")
    except Exception as e:
        print(e)

def plot_mz_histogram(df, run, ax):
    try:
        ax.hist(df['Precursor.Mz'], bins=50, color=HIST_COLOR, rwidth=0.9)
        ax.set_title(f"m/z, n = {df.height}")
    except Exception as e:
        print(e)

def plot_charge_histogram(df, run, ax):
    try:
        charges = df['Precursor.Charge'].unique().sort()
        bins = np.arange(1, charges.max() + 1.5) - 0.5
        ax.hist(df['Precursor.Charge'], bins=bins, color=HIST_COLOR, rwidth=0.9)
        ax.set_title("Precursor Charge")
        ax.set_xticks(bins + 0.5)
    except Exception as e:
        print(e)

def plot_ms1_corr_histogram(df, run, ax):
    try:
        ax.hist(df['Ms1.Profile.Corr'], bins=50, color=HIST_COLOR, rwidth=0.9)
        ax.set_title("MS1 Profile Correlation")
    except Exception as e:
        print(e)

def plot_rt_vs_irt(df, run, ax):
    try:
        irt_min = df["iRT"].quantile(0.01)
        irt_max = df["iRT"].quantile(0.99)
        frac_df = df.filter(
            (pl.col("iRT") >= irt_min) & 
            (pl.col("iRT") <= irt_max)
        )
        H, xedges, yedges = np.histogram2d(frac_df['iRT'], frac_df['RT'], bins=256)
        H = H.T
        vmax = np.percentile(H[H > 0], 99)
        ax.pcolorfast(xedges, yedges, H, cmap=CMAP, vmin=0, vmax=vmax)
        ax.set_title("RT vs library RT")
        ax.set_xlabel("library RT")
        ax.set_ylabel("RT")
    except Exception as e:
        print(e)

def plot_rt_vs_predicted_rt(df, run, ax):
    try:
        H, xedges, yedges = np.histogram2d(df['Predicted.RT'], df['RT'], bins=256)
        H = H.T
        vmax = np.percentile(H[H > 0], 99)
        ax.pcolorfast(xedges, yedges, H, cmap=CMAP, vmin=0, vmax=vmax)
        ax.set_title("RT vs Predicted.RT")
        ax.set_xlabel("Predicted.RT")
        ax.set_ylabel("RT")
    except Exception as e:
        print(e)

def plot_im_vs_predicted_im(df, run, ax):
    try:
        H, xedges, yedges = np.histogram2d(df['Predicted.IM'], df['IM'], bins=256)
        H = H.T
        vmax = np.percentile(H[H > 0], 99)
        ax.pcolorfast(xedges, yedges, H, cmap=CMAP, vmin=0, vmax=vmax)
        ax.set_title("IM vs Predicted.IM")
        ax.set_xlabel("Predicted.IM")
        ax.set_ylabel("IM")
    except Exception as e:
        print(e)

def plot_mz_vs_rt(df, run, ax):
    try:
        H, xedges, yedges = np.histogram2d(df['RT'], df['Precursor.Mz'], bins=256)
        H = H.T
        vmax = np.percentile(H[H > 0], 99)
        ax.pcolorfast(xedges, yedges, H, cmap=CMAP, vmin=0, vmax=vmax)
        ax.set_title("m/z vs RT")
        ax.set_xlabel("RT")
        ax.set_ylabel("m/z")
    except Exception as e:
        print(e)

def plot_im_vs_mz(df, run, ax):
    try:
        H, xedges, yedges = np.histogram2d(df['Precursor.Mz'], df['IM'], bins=256)
        H = H.T
        vmax = np.percentile(H[H > 0], 99)
        ax.pcolorfast(xedges, yedges, H, cmap=CMAP, vmin=0, vmax=vmax)
        ax.set_title("IM vs m/z")
        ax.set_xlabel("m/z")
        ax.set_ylabel("IM")
    except Exception as e:
        print(e)

def plot_normalization_vs_rt(df, run, ax):
    try:
        H, xedges, yedges = np.histogram2d(df['RT'], df['Normalisation.Factor'], bins=256, range=[[df['RT'].min(), df['RT'].max()], [0.0, df['Normalisation.Factor'].max() * 1.05]])
        H = H.T
        vmax = np.percentile(H[H > 0], 1)
        ax.pcolorfast(xedges, yedges, H, cmap=CMAP, vmin=0, vmax=vmax)
        ax.set_title("Normalisation Factor vs RT")
        ax.set_xlabel("RT")
        ax.set_ylabel("Normalization Factor")
    except Exception as e:
        print(e)

def plot_fwhm_vs_rt(df, run, ax):
    try:
        H, xedges, yedges = np.histogram2d(df['RT'], df['FWHM'], bins=256, range=[[df['RT'].min(), df['RT'].max()], [0.0, df['FWHM'].max()]])
        H = H.T
        vmax = np.percentile(H[H > 0], 99)
        ax.pcolorfast(xedges, yedges, H, cmap=CMAP, vmin=0, vmax=vmax)
        ax.set_title("Peak FWHM vs RT")
        ax.set_xlabel("RT")
        ax.set_ylabel("FWHM")
    except Exception as e:
        print(e)

def plot_mz_delta_vs_rt(df, run, ax):
    try:
        ms1_present = df.filter(pl.col("Ms1.Apex.Area") > 0.01).filter(pl.col("Ms1.Profile.Corr") > 0.5).filter(pl.col("Ms1.Apex.Mz.Delta") != 0.0)
        ms1_present = ms1_present.with_columns(
            (1000000.0 * (pl.col("Ms1.Apex.Mz.Delta") / pl.col("Precursor.Mz"))).alias("Mass_Delta_Ratio")
        )
        H, xedges, yedges = np.histogram2d(ms1_present['RT'], ms1_present['Mass_Delta_Ratio'], bins=256)
        H = H.T
        vmax = np.percentile(H[H > 0], 99)
        ax.pcolorfast(xedges, yedges, H, cmap=CMAP, vmin=0, vmax=vmax)
        ax.set_title("MS1 m/z delta (ppm) vs RT")
        ax.set_xlabel("RT")
        ax.set_ylabel("MS1 m/z delta (ppm)")
    except Exception as e:
        print(e)

def plot_mz_delta_vs_mz(df, run, ax):
    try:
        ms1_present = df.filter(pl.col("Ms1.Apex.Area") > 0.01).filter(pl.col("Ms1.Profile.Corr") > 0.5).filter(pl.col("Ms1.Apex.Mz.Delta") != 0.0)
        ms1_present = ms1_present.with_columns(
            (1000000.0 * (pl.col("Ms1.Apex.Mz.Delta") / pl.col("Precursor.Mz"))).alias("Mass_Delta_Ratio")
        )
        H, xedges, yedges = np.histogram2d(ms1_present['Precursor.Mz'], ms1_present['Mass_Delta_Ratio'], bins=256)
        H = H.T
        vmax = np.percentile(H[H > 0], 99)
        ax.pcolorfast(xedges, yedges, H, cmap=CMAP, vmin=0, vmax=vmax)
        ax.set_title("MS1 m/z delta (ppm) vs m/z")
        ax.set_xlabel("m/z")
        ax.set_ylabel("MS1 m/z delta (ppm)")
    except Exception as e:
        print(e)

def plot_ms2_delta_vs_rt(df, run, ax):
    try:
        selected = df.filter(pl.col("Evidence") > 3.0)
        selected = selected.with_columns(
            (1000000.0 * (pl.col("Best.Fr.Mz.Delta") / pl.col("Best.Fr.Mz"))).alias("Mass_Delta_Ratio")
        )
        H, xedges, yedges = np.histogram2d(selected['RT'], selected['Mass_Delta_Ratio'], bins=256)
        H = H.T
        vmax = np.percentile(H[H > 0], 99)
        ax.pcolorfast(xedges, yedges, H, cmap=CMAP, vmin=0, vmax=vmax)
        ax.set_title("MS2 m/z delta (ppm) vs RT")
        ax.set_xlabel("RT")
        ax.set_ylabel("MS2 m/z delta (ppm)")
    except Exception as e:
        print(e)

def plot_ms2_delta_vs_mz(df, run, ax):
    try:
        selected = df.filter(pl.col("Evidence") > 3.0)
        selected = selected.with_columns(
            (1000000.0 * (pl.col("Best.Fr.Mz.Delta") / pl.col("Best.Fr.Mz"))).alias("Mass_Delta_Ratio")
        )
        H, xedges, yedges = np.histogram2d(selected['Best.Fr.Mz'], selected['Mass_Delta_Ratio'], bins=256)
        H = H.T
        vmax = np.percentile(H[H > 0], 99)
        ax.pcolorfast(xedges, yedges, H, cmap=CMAP, vmin=0, vmax=vmax)
        ax.set_title("MS2 m/z delta (ppm) vs m/z")
        ax.set_xlabel("m/z")
        ax.set_ylabel("MS2 m/z delta (ppm)")
    except Exception as e:
        print(e)

def plot_identified_ms1_tic(df, run, ax):
    try:
        rt_min, rt_max = df['RT'].min(), df['RT'].max()
        rt_bins = np.linspace(rt_min, rt_max, 128 + 1)
        df = df.with_columns(pl.col("RT").cut(rt_bins).alias("bin"))
        rt_area = df.group_by("bin").agg(pl.sum("Ms1.Apex.Area"))
        rt_area = rt_area.unique(subset=["bin"]).sort("bin")
        ax.bar(rt_area["bin"], rt_area["Ms1.Apex.Area"], color=HIST_COLOR)
        ax.set_title("Identified MS1 TIC")
        ax.set_xlabel("RT")
        ax.set_ylabel("Total MS1 signal at apex")
        ax.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
    except Exception as e:
        print(e)

def plot_precursors_per_protein(df, run, ax):
    try:
        sig_proteins = df.filter(pl.col("PG.Q.Value") <= 0.01)
        pr_per_prot = sig_proteins.group_by("Protein.Group").agg(pl.count("Precursor.Id")).get_column("Precursor.Id").value_counts().sort("Precursor.Id")
        max_count = pr_per_prot['Precursor.Id'].max()
        x_values = list(range(1, max_count + 1))
        y_values = [0] * max_count
        for row in pr_per_prot.iter_rows():
            y_values[row[0] - 1] = row[1]
        ax.bar(x_values, y_values, color=HIST_COLOR)
        num_proteins = sig_proteins.select(pl.col("Protein.Group").n_unique()).item()
        ax.set_title(f"Precursors per protein group, n = {num_proteins}")
    except Exception as e:
        print(e)

def plot_fwhm_histogram(df, run, ax):
    try:
        ax.hist(df['FWHM'], bins=50, color=HIST_COLOR, rwidth=0.9, range=[0.0, df['FWHM'].max()])
        ax.set_title("Peak Width (FWHM)")
    except Exception as e:
        print(e)

def plot_qvalue_histogram(df, run, ax):
    try:
        ax.hist(df['Q.Value'], bins=50, color=HIST_COLOR, rwidth=0.9)
        ax.set_title("Q-Value")
    except Exception as e:
        print(e)

def plot_irt_proportion(df, run, ax):
    try:
        global run_to_proportions, irt_bin_centers
        irt_proportions = run_to_proportions[run]['irt_proportions']
        proportions = irt_proportions['proportion'].to_numpy()
        ax.bar(range(len(proportions)), proportions, color=HIST_COLOR)
        ax.set_title("ID rate per library RT Bin")
        ax.set_xlabel("library RT")
        ax.set_ylabel("Proportion")
        ax.set_ylim(0, 1)
        tick_positions = np.arange(0, 50, 8)
        tick_labels = [f"{irt_bin_centers[i]:.1f}" for i in tick_positions]
        ax.set_xticks(tick_positions)
        ax.set_xticklabels(tick_labels)
    except Exception as e:
        print(e)

def plot_mz_proportion_global(df, run, ax):
    try:
        global run_to_proportions, mz_bin_centers
        mz_proportions = run_to_proportions[run]['mz_proportions']
        proportions = mz_proportions['proportion'].to_numpy()
        ax.bar(range(len(proportions)), proportions, color=HIST_COLOR)
        ax.set_title("ID rate per m/z Bin")
        ax.set_xlabel("m/z")
        ax.set_ylabel("Proportion")
        ax.set_ylim(0, 1)
        
        cap_mz = max(mz_bin_centers)
        min_mz = np.floor(min(mz_bin_centers) / 100) * 100  # Round down to nearest 100
        max_mz = np.ceil(cap_mz / 100) * 100   # Round up to nearest 100
        step_size = 100 if max_mz - min_mz <= 1000 else 200  # Adjust step based on range
        
        mz_ticks = np.arange(min_mz, max_mz + step_size, step_size)
        flt = []
        for tick in mz_ticks:
            if (tick <= cap_mz + 20): flt.append(True)
            else: flt.append(False)
        mz_ticks = mz_ticks[flt]
        tick_positions = []
        tick_labels = []
        
        for mz in mz_ticks:
            bin_idx = np.argmin(np.abs(mz_bin_centers - mz))
            if bin_idx not in tick_positions:
                tick_positions.append(bin_idx)
                tick_labels.append(f"{int(mz)}")
        
        ax.set_xticks(tick_positions)
        ax.set_xticklabels(tick_labels)
    except Exception as e:
        print(e)

def plot_page(df, run):
    """Generate a dynamic grid of plots for a single acquisition."""
    enabled_plots = [plot for plot in plot_configs if plot_configs[plot]['enabled']]
    N = len(enabled_plots)
    cols = 4
    rows = (N + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(20, 5 * rows))
    axes = axes.flat if N > 1 else [axes]

    for ax, plot in zip(axes, enabled_plots):
        globals()[plot_configs[plot]['function']](df, run, ax)

    for ax in axes[N:]:
        ax.axis('off')

    fig.suptitle(f"{run}", fontsize=16, y=0.98)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    return fig

def compute_summary_metrics(df):
    """Compute summary metrics for each run, ordered by Run.Index if available."""
    if 'Run.Index' in df.columns:
        df_sorted = df.sort('Run.Index')
        run_order = df.select(['Run', 'Run.Index']).unique().sort(by='Run.Index')
        runs = run_order['Run'].to_list()
    else:
        df_sorted = df
        runs = df.select('Run').unique(maintain_order=True).to_series().to_list()

    metrics_df = df_sorted.group_by('Run', maintain_order=True).agg([
        pl.len().alias('num_precursors'),
        (pl.col('Protein.Group').filter(pl.col('PG.Q.Value') <= 0.01).n_unique()).alias('num_proteins'),
        pl.col('Ms1.Apex.Area').sum().alias('total_ms1_tic')
    ])

    metrics = {col: metrics_df[col].to_list() for col in metrics_df.columns if col != 'Run'}
    return runs, metrics

def plot_summary(runs, metrics, pdf):
    metric_display_names = {
        'num_precursors': 'Number of Identified Precursors',
        'num_proteins': 'Number of Identified Protein Groups',
        'total_ms1_tic': 'Total MS1 Apex Signal'
    }

    N = len(runs)
    fig_width = 10
    L = max([len(run) for run in runs]) if runs else 1

    # font size
    if N > 1:
        S = 14
        S_N = (fig_width / (N - 1)) * 72 * 0.35
        S_L = (fig_width * 72) / (1.5 * L)
        if S_N < S: S = S_N
        if S_L < S: S = S_L 
    else:
        S = 5

    for metric_name, metric_values in metrics.items():
        fig, ax = plt.subplots(figsize=(fig_width, 6), constrained_layout=True)
        ax.plot(range(N), metric_values, 'o-', color='darkblue')
        display_name = metric_display_names.get(metric_name, metric_name)
        ax.set_title(f"{display_name} vs Run Order")
        ax.set_xlabel("Run")
        ax.set_ylabel(display_name)
        ax.grid(True, linestyle='--', alpha=0.7)
        ax.set_xticks(range(N))
        ax.set_xticklabels(runs, rotation=45, ha='right', fontsize=S)
        pdf.savefig(fig)
        plt.close(fig)
        
def main(parquet_file):
    global run_to_proportions, irt_bin_centers, mz_bin_centers
    df = pl.read_parquet(parquet_file).filter(pl.col('Q.Value') <= 0.01)

    precursor_run_counts = df.group_by('Precursor.Id').agg(pl.col('Run').n_unique().alias('run_count'))
    max_run_count = precursor_run_counts['run_count'].max()
    threshold = int(np.ceil(0.99 * max_run_count))

    # Precursors with high data completeness
    selected_precursors = precursor_run_counts.filter(pl.col('run_count') >= threshold)['Precursor.Id'].to_list()
    
    # Global-confident precursors
    global_precursors = df.filter(pl.col('Global.Q.Value') <= 0.01)['Precursor.Id'].to_list()

    # Sort by run index
    if 'Run.Index' in df.columns:
        df_sorted = df.sort('Run.Index')
        run_order = df.select(['Run', 'Run.Index']).unique().sort(by='Run.Index')
        runs = run_order['Run'].to_list()
    else:
        df_sorted = df
        runs = df.select('Run').unique(maintain_order=True).to_series().to_list()

    runs, metrics = compute_summary_metrics(df)

    selected_df = df_sorted.filter(pl.col('Precursor.Id').is_in(selected_precursors))
    agg_list = [
        pl.col('RT').quantile(0.25).alias('RT_Q1'),
        pl.col('RT').quantile(0.5).alias('RT_median'),
        pl.col('RT').quantile(0.75).alias('RT_Q3'),
    ]
    if df['IM'].max() > 0.5:
        agg_list.extend([
            pl.col('IM').quantile(0.25).alias('IM_Q1'),
            pl.col('IM').quantile(0.5).alias('IM_median'),
            pl.col('IM').quantile(0.75).alias('IM_Q3'),
        ])
    agg_list.extend([
        pl.col('FWHM').quantile(0.25).alias('FWHM_Q1'),
        pl.col('FWHM').quantile(0.5).alias('FWHM_median'),
        pl.col('FWHM').quantile(0.75).alias('FWHM_Q3'),
    ])
    stats_df = selected_df.group_by('Run', maintain_order=True).agg(agg_list)

    frac_df = df.filter(pl.col('Precursor.Id').is_in(global_precursors))
    irt_min = frac_df["iRT"].quantile(0.01)-0.00001
    irt_max = frac_df["iRT"].quantile(0.99)+0.00001
    frac_df = frac_df.filter(pl.col('Precursor.Id').is_in(global_precursors)).filter(
        (pl.col("iRT") >= irt_min) & 
        (pl.col("iRT") <= irt_max)
    )
    mz_min, mz_max = frac_df['Precursor.Mz'].min()-0.00001, frac_df['Precursor.Mz'].max()+0.00001
    irt_bins = np.linspace(irt_min, irt_max, 51)
    mz_bins = np.linspace(mz_min, mz_max, 51)

    precursor_info = frac_df.group_by('Precursor.Id').agg(
        pl.col('iRT').first().alias('iRT'),
        pl.col('Precursor.Mz').first().alias('MZ')
    ).with_columns(
        pl.col('iRT').cut(irt_bins, labels=[str(i) for i in range(52)]).alias('IRT_bin'),
        pl.col('MZ').cut(mz_bins, labels=[str(i) for i in range(52)]).alias('MZ_bin')
    )

    precursor_info = precursor_info.filter(pl.col('Precursor.Id').is_in(global_precursors))

    total_irt_counts = precursor_info.group_by('IRT_bin').agg(pl.len().alias('total_count')).sort('IRT_bin').with_columns(pl.col('IRT_bin').cast(pl.Utf8)).with_columns(pl.col("total_count").replace(0, 1))
    total_mz_counts = precursor_info.group_by('MZ_bin').agg(pl.len().alias('total_count')).sort('MZ_bin').with_columns(pl.col('MZ_bin').cast(pl.Utf8)).with_columns(pl.col("total_count").replace(0, 1))

    run_to_proportions = {}
    for run in runs:
        run_data = frac_df.filter(pl.col("Run") == run)
        run_data = run_data.join(precursor_info.select(['Precursor.Id', 'IRT_bin', 'MZ_bin']), on='Precursor.Id')
        
        irt_counts = run_data.group_by('IRT_bin').agg(pl.len().alias('count')).sort('IRT_bin').with_columns(pl.col('IRT_bin').cast(pl.Utf8))
        all_irt_bins = pl.DataFrame({'IRT_bin': [str(i) for i in range(1,51)]})
        irt_counts = all_irt_bins.join(irt_counts, on='IRT_bin', how='left').fill_null(0)
        irt_proportions = irt_counts.join(total_irt_counts, on='IRT_bin').with_columns(
            (pl.col('count') / pl.col('total_count')).alias('proportion')
        )
        
        mz_counts = run_data.group_by('MZ_bin').agg(pl.len().alias('count')).sort('MZ_bin').with_columns(pl.col('MZ_bin').cast(pl.Utf8))
        all_mz_bins = pl.DataFrame({'MZ_bin': [str(i) for i in range(1,51)]})
        mz_counts = all_mz_bins.join(mz_counts, on='MZ_bin', how='left').fill_null(0)
        mz_proportions = mz_counts.join(total_mz_counts, on='MZ_bin').with_columns(
            (pl.col('count') / pl.col('total_count')).alias('proportion')
        )
        
        run_to_proportions[run] = {'irt_proportions': irt_proportions, 'mz_proportions': mz_proportions}

    irt_bin_centers = (irt_bins[:-1] + irt_bins[1:]) / 2
    mz_bin_centers = (mz_bins[:-1] + mz_bins[1:]) / 2

    # Per-acquisition PDF
    with PdfPages(f"{parquet_file.rsplit('.', 1)[0]}_runs.pdf") as pdf:
        for run in runs:
            run_data = df.filter(pl.col("Run") == run)
            fig = plot_page(run_data, run)
            pdf.savefig(fig)
            plt.close(fig)

    # Trends PDF
    with PdfPages(f"{parquet_file.rsplit('.', 1)[0]}_trends.pdf") as pdf:
        plot_summary(runs, metrics, pdf)

        metrics_to_plot = ['RT']
        if df['IM'].max() > 0.5:
            metrics_to_plot.append('IM')
        metrics_to_plot.append('FWHM')

        N = len(runs)
        fig_width = 10
        L = max([len(run) for run in runs]) if runs else 1
        if N > 1:
            S = 14
            S_N = (fig_width / (N - 1)) * 72 * 0.35
            S_L = (fig_width * 72) / (1.5 * L)
            if S_N < S: S = S_N
            if S_L < S: S = S_L 
        else:
            S = 5

        for metric in metrics_to_plot:
            fig, ax = plt.subplots(figsize=(fig_width, 6), constrained_layout=True)
            ax.plot(range(N), stats_df[f'{metric}_Q3'], label='Q3', color='steelblue')
            ax.plot(range(N), stats_df[f'{metric}_median'], label='Median', color='black')
            ax.plot(range(N), stats_df[f'{metric}_Q1'], label='Q1', color='seagreen')
            ax.set_title(f"{metric} Distribution for Consistently Identified Precursors")
            ax.set_xlabel("Run")
            ax.set_ylabel(metric)
            ax.legend()
            ax.grid(True, linestyle='--', alpha=0.7)
            ax.set_xticks(range(N))
            ax.set_xticklabels(runs, rotation=45, ha='right', fontsize=S)
            pdf.savefig(fig)
            plt.close(fig)

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print("Usage: diann-stats.py <path_to_parquet_file>")
        sys.exit(1)
    main(sys.argv[1])
