import logging
import argparse
import os
import json
from tqdm import tqdm
import pandas as pd
import plotly.express as px
import torch
import numpy as np
import warnings
import plotly.graph_objects as go
from plotly.colors import sample_colorscale
import re

warnings.filterwarnings("ignore")

# mapping between langs and ids
LANG2ID = {
    "unknown": -1,  # unknown
    "eng-train": 0,  # English
    "cmn-train": 1,  # Chinese
    "deu-train": 2,  # German
    "spa-train": 3,  # Spanish
    "jpn-train": 4,  # Japanese
    "fra-train": 5,  # French
    "ita-train": 6,  # Italian
    "por-train": 7,  # Portuguese
    "ind-train": 8,  # Indonesian
}

# reverse mapping
ID2LANG = {v: k for k, v in LANG2ID.items()}

LANG_SCAL = {
    "eng-train": 0.2,  # English
    "cmn-train": 0.2,  # Chinese
    "deu-train": 0.2,  # German
    "spa-train": 0.2,  # Spanish
    "jpn-train": 0.2,  # Japanese
    "fra-train": 0.2,  # French
    "ita-train": 0.2,  # Italian
    "por-train": 0.2,  # Portuguese
    "ind-train": 0.2,  # Indonesian
}

LANG_SCAL = {
    "eng-train": 0.125,  # English
    "cmn-train": 0.125,  # Chinese
    "deu-train": 0.125,  # German
    "spa-train": 0.125,  # Spanish
    "jpn-train": 0.125,  # Japanese
    "fra-train": 0.125,  # French
    "ita-train": 0.125,  # Italian
    "por-train": 0.125,  # Portuguese
    "ind-train": 0.125,  # Indonesian
}

def get_layer_sublayers(i):
    return [f'model.layers.{i}.self_attn.q_proj.weight',
            f'model.layers.{i}.self_attn.k_proj.weight',
            f'model.layers.{i}.self_attn.v_proj.weight',
            f'model.layers.{i}.self_attn.o_proj.weight',
            f'model.layers.{i}.mlp.gate_proj.weight',
            f'model.layers.{i}.mlp.up_proj.weight',
            f'model.layers.{i}.mlp.down_proj.weight',
            f'model.layers.{i}.input_layernorm.weight',
            f'model.layers.{i}.post_attention_layernorm.weight']

threestep2layer = {
    "embeddings_input": ['model.embed_tokens.weight'],
    "layer 0-7": sum([get_layer_sublayers(i) for i in range(0, 8)], []),
    "layer 8-15": sum([get_layer_sublayers(i) for i in range(8, 16)], []),
    "layer 16-23": sum([get_layer_sublayers(i) for i in range(16, 24)], []),
    "unembeddings_output": ['model.norm.weight', 'lm_head.weight']
}

layer2threestep = dict([
    (sublayer, layer) for layer, sublayers in threestep2layer.items() for sublayer in sublayers
])


# LAYER_ORDER = ['lm.transformer.wte.weight', 'lm.transformer.wpe.weight', 'lm.transformer.h.0.ln_1.weight', 'lm.transformer.h.0.ln_1.bias', 'lm.transformer.h.0.attn.c_attn.weight', 'lm.transformer.h.0.attn.c_attn.bias', 'lm.transformer.h.0.attn.c_proj.weight', 'lm.transformer.h.0.attn.c_proj.bias', 'lm.transformer.h.0.ln_2.weight', 'lm.transformer.h.0.ln_2.bias', 'lm.transformer.h.0.mlp.c_fc.weight', 'lm.transformer.h.0.mlp.c_fc.bias', 'lm.transformer.h.0.mlp.c_proj.weight', 'lm.transformer.h.0.mlp.c_proj.bias', 'lm.transformer.h.1.ln_1.weight', 'lm.transformer.h.1.ln_1.bias', 'lm.transformer.h.1.attn.c_attn.weight', 'lm.transformer.h.1.attn.c_attn.bias', 'lm.transformer.h.1.attn.c_proj.weight', 'lm.transformer.h.1.attn.c_proj.bias', 'lm.transformer.h.1.ln_2.weight', 'lm.transformer.h.1.ln_2.bias', 'lm.transformer.h.1.mlp.c_fc.weight', 'lm.transformer.h.1.mlp.c_fc.bias', 'lm.transformer.h.1.mlp.c_proj.weight', 'lm.transformer.h.1.mlp.c_proj.bias', 'lm.transformer.h.2.ln_1.weight', 'lm.transformer.h.2.ln_1.bias', 'lm.transformer.h.2.attn.c_attn.weight', 'lm.transformer.h.2.attn.c_attn.bias', 'lm.transformer.h.2.attn.c_proj.weight', 'lm.transformer.h.2.attn.c_proj.bias', 'lm.transformer.h.2.ln_2.weight', 'lm.transformer.h.2.ln_2.bias', 'lm.transformer.h.2.mlp.c_fc.weight', 'lm.transformer.h.2.mlp.c_fc.bias', 'lm.transformer.h.2.mlp.c_proj.weight', 'lm.transformer.h.2.mlp.c_proj.bias', 'lm.transformer.h.3.ln_1.weight', 'lm.transformer.h.3.ln_1.bias', 'lm.transformer.h.3.attn.c_attn.weight', 'lm.transformer.h.3.attn.c_attn.bias', 'lm.transformer.h.3.attn.c_proj.weight', 'lm.transformer.h.3.attn.c_proj.bias', 'lm.transformer.h.3.ln_2.weight', 'lm.transformer.h.3.ln_2.bias', 'lm.transformer.h.3.mlp.c_fc.weight', 'lm.transformer.h.3.mlp.c_fc.bias', 'lm.transformer.h.3.mlp.c_proj.weight', 'lm.transformer.h.3.mlp.c_proj.bias', 'lm.transformer.h.4.ln_1.weight', 'lm.transformer.h.4.ln_1.bias', 'lm.transformer.h.4.attn.c_attn.weight', 'lm.transformer.h.4.attn.c_attn.bias', 'lm.transformer.h.4.attn.c_proj.weight', 'lm.transformer.h.4.attn.c_proj.bias', 'lm.transformer.h.4.ln_2.weight', 'lm.transformer.h.4.ln_2.bias', 'lm.transformer.h.4.mlp.c_fc.weight', 'lm.transformer.h.4.mlp.c_fc.bias', 'lm.transformer.h.4.mlp.c_proj.weight', 'lm.transformer.h.4.mlp.c_proj.bias', 'lm.transformer.h.5.ln_1.weight', 'lm.transformer.h.5.ln_1.bias', 'lm.transformer.h.5.attn.c_attn.weight', 'lm.transformer.h.5.attn.c_attn.bias', 'lm.transformer.h.5.attn.c_proj.weight', 'lm.transformer.h.5.attn.c_proj.bias', 'lm.transformer.h.5.ln_2.weight', 'lm.transformer.h.5.ln_2.bias', 'lm.transformer.h.5.mlp.c_fc.weight', 'lm.transformer.h.5.mlp.c_fc.bias', 'lm.transformer.h.5.mlp.c_proj.weight', 'lm.transformer.h.5.mlp.c_proj.bias', 'lm.transformer.h.6.ln_1.weight', 'lm.transformer.h.6.ln_1.bias', 'lm.transformer.h.6.attn.c_attn.weight', 'lm.transformer.h.6.attn.c_attn.bias', 'lm.transformer.h.6.attn.c_proj.weight', 'lm.transformer.h.6.attn.c_proj.bias', 'lm.transformer.h.6.ln_2.weight', 'lm.transformer.h.6.ln_2.bias', 'lm.transformer.h.6.mlp.c_fc.weight', 'lm.transformer.h.6.mlp.c_fc.bias', 'lm.transformer.h.6.mlp.c_proj.weight', 'lm.transformer.h.6.mlp.c_proj.bias', 'lm.transformer.h.7.ln_1.weight', 'lm.transformer.h.7.ln_1.bias', 'lm.transformer.h.7.attn.c_attn.weight', 'lm.transformer.h.7.attn.c_attn.bias', 'lm.transformer.h.7.attn.c_proj.weight', 'lm.transformer.h.7.attn.c_proj.bias', 'lm.transformer.h.7.ln_2.weight', 'lm.transformer.h.7.ln_2.bias', 'lm.transformer.h.7.mlp.c_fc.weight', 'lm.transformer.h.7.mlp.c_fc.bias', 'lm.transformer.h.7.mlp.c_proj.weight', 'lm.transformer.h.7.mlp.c_proj.bias', 'lm.transformer.ln_f.weight', 'lm.transformer.ln_f.bias', 'lm.lm_head.weight']
# BLOCK_ORDER = ["embeddings", "pos_embeddings"] + [f"block_{i}" for i in range(8)] + ["ln_f"]

# gpt2_layer_mapping = {
#     "embeddings" : ["lm.transformer.wte.weight", "lm.lm_head.weight"],
#     "pos_embeddings": ["lm.transformer.wpe.weight"],
#     "ln_f": ["lm.transformer.ln_f.weight", "lm.transformer.ln_f.bias"],
# }

# num_layers = 8

# for i in range(num_layers):
#     gpt2_layer_mapping[f"block_{i}"] = [
#         f"lm.transformer.h.{i}.attn.c_attn.weight",
#         f"lm.transformer.h.{i}.attn.c_attn.bias",
#         f"lm.transformer.h.{i}.attn.c_proj.weight",
#         f"lm.transformer.h.{i}.attn.c_proj.bias",
#         f"lm.transformer.h.{i}.ln_1.weight",
#         f"lm.transformer.h.{i}.ln_1.bias",
#         f"lm.transformer.h.{i}.mlp.c_fc.weight",
#         f"lm.transformer.h.{i}.mlp.c_fc.bias",
#         f"lm.transformer.h.{i}.mlp.c_proj.weight",
#         f"lm.transformer.h.{i}.mlp.c_proj.bias",
#         f"lm.transformer.h.{i}.ln_2.weight",
#         f"lm.transformer.h.{i}.ln_2.bias",
#     ]

def plot_scores(df, score_name):
    fig = px.line(df, 
                  x='step', 
                  y='score', 
                  color='type',
                  title=f'{score_name} over training', markers=True)
    fig.update_layout(
        xaxis_title='step',
        yaxis_title=score_name.capitalize(),
        legend_title_text='Data partition',
        template='plotly_white'
    )
    # no markers
    fig.update_traces(mode='lines')

    return fig

def store_figure(fig, pname):
    # store figure in all kinds of formats (html, png, png high res, png low res, pdf)
    # fig.write_image(f"{pname}.png", scale=1)
    fig.write_image(f"{pname}_high_res.png", scale=2)
    # fig.write_image(f"{pname}_low_res.png", scale=0.5)
    fig.write_image(f"{pname}.pdf")
    fig.write_html(f"{pname}.html")


def make_timeline(df, plotdir, score_name='sum', test_partition=None, train_partition=None, group_by="depth"):
    # sort df by step

    df = df.sort_values(by=['step', 'layer_depth'])
    df_scale = df.copy()

    print(df.head())

    if group_by == "depth":
        df["group"] =  df["layer"].apply(lambda x: "layer " + re.search(r'layers\.(\d+)', x).group(1) if re.search(r'layers\.(\d+)', x) else x.split('.')[1])
        # df["group"][1:] = df["group"][1:] // 9 + 1
        df = df.groupby(["step", "group"]).agg({
            score_name: "sum",
            # extract "layer.x" from layer names and take first
            "layer_depth": "first",
            "layer": lambda x: "layer " + re.search(r'layers\.(\d+)', x.iloc[0]).group(1) if re.search(r'layers\.(\d+)', x.iloc[0]) else x.iloc[0].split('.')[1]
        }).reset_index()
        print("Grouped by layer:")
        print(df['layer'].unique())

    elif group_by == "type":
        print(df["layer"].unique())
        # strip leading integers
        df["group"] = df["layer"].apply(lambda x: ".".join(x.split('.')[-3:]).lstrip("0123456789").lstrip("."))
        df = df.groupby(["step", "group"]).agg({
            score_name: "sum",
            "layer_depth": "mean",
            "layer": lambda x: ".".join(x.iloc[0].split('.')[-3:-1]).lstrip("0123456789").lstrip(".")
        }).reset_index().sort_values(by=['step', 'layer_depth'])
        df["layer_depth"] = df["layer_depth"].apply(lambda x: int(x))
        print("Grouped by type:")
        print(df['layer'].unique())

    elif group_by == "threesteps":
        df["group"] = df["layer"].apply(lambda x: layer2threestep.get(x, x))
        df = df.groupby(["step", "group"]).agg({
            score_name: "sum",
            "layer_depth": "mean",
            "layer": lambda x: layer2threestep.get(x.iloc[0], x.iloc[0])
        }).reset_index().sort_values(by=['step', 'layer_depth'])
        df["layer_depth"] = df["layer_depth"].apply(lambda x: int(x))
        print("Grouped by three steps:")
        print(df['layer'].unique())


    print(df.head())

    # rgb = px.colors.convert_colors_to_same_type(px.colors.sequential.RdBu)[0]

    # colorscale = []
    # n_steps = 4  # Control the number of colors in the final colorscale
    # for i in range(len(rgb) - 1):
    #     for step in np.linspace(0, 1, n_steps):
    #         colorscale.append(px.colors.find_intermediate_color(rgb[i], rgb[i + 1], step, colortype='rgb'))


    # create a plotly figure
    # fig = px.line(df, 
    #               x='step', 
    #               y='sum', 
    #               color='layer_depth',
    #               color_discrete_sequence=colorscale,
    #               title=f'Accumulated Scores Timeline',
    #               markers=True)

    layer_order = list(df_scale['layer'].unique())
    # print(layer_order)
    layer_ids = [i / len(layer_order) for i in range(0, len(layer_order))]

    depth_color_mapping = sample_colorscale('Viridis', layer_ids)

    # thank you, gpt
    type_color_mapping = {
        # === LayerNorms (clearly separated neutrals) ===
        "input_layernorm": "#D9D9D9",            # very light gray
        "post_attention_layernorm": "#969696",   # mid gray
        "model.norm": "#252525",                 # near-black

        # === Embedding & LM head (distinct, unique hues) ===
        "model.embed_tokens": "#1B9E77",         # strong teal
        "lm_head": "#6A3D9A",                    # deep purple

        # === MLP (high-contrast warm colors) ===
        "mlp.up_proj": "#FDBF6F",                # light amber
        "mlp.gate_proj": "#FF7F00",              # saturated orange
        "mlp.down_proj": "#E31A1C",              # strong red

        # === Self-attention (widely spaced blues) ===
        "self_attn.q_proj": "#08306B",            # very dark navy
        "self_attn.k_proj": "#2171B5",            # strong blue
        "self_attn.v_proj": "#6BAED6",            # light blue
        "self_attn.o_proj": "#00B4FF",            # cyan-blue
    }

    dotted = ['input_layernorm', 'post_attention_layernorm', 'model.norm']
    striped = ['model.embed_tokens', 'lm_head']

    data = []
    for layer in df['layer'].unique():
        layer_df = df[df['layer'] == layer]
        # print(layer_df['layer_depth'])

        if group_by == "type":
            color = type_color_mapping.get(layer_df['layer'].iloc[0], depth_color_mapping[layer_df['layer_depth'].iloc[0]])
            data.append(go.Scatter(
                x=layer_df['step'],
                y=layer_df[score_name],  
                mode='lines',
                name=layer,
                # thicker lines
                line=dict(color=color, width=2),
                # apply dots/stripes for certain types
                # marker=dict(size=5),
                hovertemplate=f"Type: {layer}<br>step: %{{x}}<br>{score_name.capitalize()}: %{{y}}<extra></extra>"
            ))
            data[-1].update(line=dict(
                dash='dot' if layer in dotted else 'solid' if layer not in striped else 'dash'
            ))
        else:
             data.append(go.Scatter(
                x=layer_df['step'],
                y=layer_df[score_name],  
                mode='lines',
                name=layer,
                line=dict(color=depth_color_mapping[layer_df['layer_depth'].iloc[0]]),
                # marker=dict(size=5),
                hovertemplate=f"Layer: {layer}<br>step: %{{x}}<br>{score_name.capitalize()}: %{{y}}<extra></extra>"
            ))
            

    fig = go.Figure(data=data)
    fig.update_yaxes(exponentformat = 'power')
   
    # log y axis
    # fig.update_yaxes(type='log')
    
    # set the color scale to be the layer depth
    # fig.update_traces(line=dict(color=df['layer_depth'], colorscale='Viridis'))

    if train_partition.isdigit():
        train_partition = ID2LANG[int(train_partition)]
    
    print(train_partition, test_partition)

    fig.update_layout(
        xaxis_title='<b>Step</b>',
        yaxis_title='<b>Accumulated score</b>',
        # legend_title_text='<b>Layer Depth</b>',
        template='plotly_white',
        title=f'Train: {train_partition} Test: {test_partition} - {score_name.capitalize()}',
    )

    fig.update_layout(
        font_family="Times New Roman",
        title_font_size=32,
        xaxis_title_font_size=24,
        yaxis_title_font_size=24,
        # tick_font_size=18,
    )

    # enlarge legend
    fig.update_layout(
        legend=dict(
            font_size=16,
            title_font_size=24,
        )
    )

    fig.update_layout(
        xaxis = dict(
            tickmode = 'array',
            tickvals = list(range(0, df['step'].max()+1, 10000)),
            # make XXk style strings
            ticktext = [f"{i//1000}k" for i in range(0, df['step'].max()+1, 10000)],
            tickfont_size=16,
            )
    )
    fig.update_yaxes(tickfont_size=16)
    return fig


def get_parser_args():
    parser = argparse.ArgumentParser()
    # parser.add_argument("--dir", default="/nfs/brick/explainables/runs/full_bs512_ga32_shuffled_10kwarmup_explaind_scores_corrected/", type=str, help="Dir of training checkpoints.")
    parser.add_argument("--dir", default="/nfs/brick/explainables/runs/full_bs512_ga32_shuffled_10kwarmup_explaind_scores/wlt/parallel_train_final/")
    parser.add_argument("--plotdir", default="plots2/wlt/parallel_train/", type=str, help="Dir to save plots.")
    parser.add_argument("--diffs", action='store_true', help="Whether to plot differences between good and bad sentence loss.")
    parser.add_argument("--apply_lang_scaling", action='store_true', help="Whether to apply language scaling to the scores.")
    parser.add_argument("--grouping", default="depth", type=str, help="Whether and which grouping to apply to model parameters for plotting [layers/type/none].")
    parser.add_argument("--plot_stats", action='store_true', help="Whether to plot mean and std as well (slow).")
    parser.add_argument("--debug", action='store_true', help="Whether to run in debug mode with fewer checkpoints.")
    parser.add_argument("--normalized", action='store_true', help="Whether to normalize scores per step and per plot.")
    args = parser.parse_args()
    return args

def main():
    args = get_parser_args()
    checkpoint_path = args.dir
    plotdir = args.plotdir
    plot_stats = args.plot_stats

    if not os.path.exists(plotdir):
        logging.info(f"Creating plot directory: {plotdir}")
        os.makedirs(plotdir)

    checkpoints_paths = [f for f in os.listdir(checkpoint_path) if f.startswith("checkpoint")]

    print(f"Found {len(checkpoints_paths)} checkpoints in {checkpoint_path}")

    checkpoints = dict()
    layer_checkpoints = dict()
    
    if args.debug:
        checkpoints_paths = checkpoints_paths[:5]
        print("Debug mode: only processing first 5 checkpoints")

    for p in tqdm(checkpoints_paths, desc="Loading checkpoints"):
        lpath = os.path.join(checkpoint_path, p, 'accumulated_scores.pt')
        llpath = os.path.join(checkpoint_path, p, 'accumulated_layer_scores.pt')
        if os.path.exists(lpath):
            checkpoint_step = int(p.split('-')[-1].split('_')[0])
            checkpoints[checkpoint_step] = torch.load(lpath, map_location='cpu')
        else:
            print(lpath, "does not exist")
        if os.path.exists(llpath):
            layer_checkpoints[checkpoint_step] = torch.load(llpath, map_location='cpu')
            
    # make a df out of it
    data = {
        'step': [],
        'sum': [],
        'mean': [],
        'std': [],
        'num_vals': [],
        'layer': [],
        'train_partition': [],
        'test_partition': [],
        'correct': []
    }
    # layer_data = {
    #     'step': [],
    #     'sum': [],
    #     'layer': [],
    #     'train_partition': [],
    #     'test_partition': [],
    #     'correct': []
    # }

    for step, checkpoint in tqdm(checkpoints.items(), desc="Processing checkpoints into df"):
        for stype, checkpoint_type in checkpoint.items():
            for par, param_score in checkpoint_type.items():
                if par == 'num_vals':
                    continue
                data['step'].append(step)
                data['sum'].append(param_score["sum"])
                data['mean'].append(param_score["mean"] / checkpoint_type["num_vals"])
                data['std'].append(param_score["std"] / checkpoint_type["num_vals"])
                data['num_vals'].append(checkpoint_type["num_vals"])
                data['layer'].append(par)
                data['train_partition'].append(stype[0])
                data['test_partition'].append(stype[1])
                data['correct'].append(stype[2])
                
    
    # for step, layer_checkpoint in tqdm(layer_checkpoints.items(), desc="Processing layer checkpoints into df"):
    #     for stype, checkpoint_type in layer_checkpoint.items():
    #         for layer, layer_score in checkpoint_type.items():
    #             layer_data['step'].append(step)
    #             layer_data['sum'].append(layer_score["sum"])
    #             layer_data['layer'].append(layer)
    #             layer_data['train_partition'].append(stype[0])
    #             layer_data['test_partition'].append(stype[1])
    #             layer_data['correct'].append(stype[2])

    df = pd.DataFrame(data)
    # layer_df = pd.DataFrame(layer_data)

    # store df at checkpoint_path
    df.to_csv(os.path.join(checkpoint_path, 'accumulated_scores.csv'), index=False)
    # layer_df.to_csv(os.path.join(checkpoint_path, 'accumulated_scores_by_layer.csv'), index=False)

    # ignore incorrect for now
    # df = df[df['correct'] == 'correct']
    # layer_df = layer_df[layer_df['correct'] == 'correct']

    df['layer_depth'] = 1 # df['layer'].apply(lambda x: LAYER_ORDER.index(x))
    layer_depth_mapping = dict([(parname, i) for i, parname in enumerate(df['layer'].unique())])
    df['layer_depth'] = df['layer'].apply(lambda x: layer_depth_mapping[x])
    # print(df['layer_depth'].head())
    # layer_df['layer_depth'] = layer_df['layer'].apply(lambda x: BLOCK_ORDER.index(x))

    # print(layer_df['layer'].head())

    # plot each type seperately
    for corr in df['correct'].unique():
        for train_partition in df['train_partition'].unique():
            for test_partition in df['test_partition'].unique():
                print(train_partition, test_partition, corr)
                if args.diffs:
                    if test_partition.endswith("_bad"):
                        continue
                    # find corresponding bad test partition
                    bad_test_partition = test_partition.replace("_good", "_bad")
                    filtered_df_good = df[(df['train_partition'] == train_partition) & 
                                          (df['test_partition'] == test_partition) &
                                          (df['correct'] == corr)]
                    filtered_df_bad = df[(df['train_partition'] == train_partition) &
                                            (df['test_partition'] == bad_test_partition) &
                                            (df['correct'] == corr)]
                    # merge on step and layer
                    filtered_df = pd.merge(filtered_df_good, filtered_df_bad, on=['step', 'layer', 'train_partition', 'correct'], suffixes=('_good', '_bad'))
                    # compute difference
                    filtered_df['sum'] = filtered_df['sum_good'] - filtered_df['sum_bad']
                    filtered_df['mean'] = filtered_df['mean_good'] - filtered_df['mean_bad']
                    filtered_df['std'] = filtered_df['std_good'] + filtered_df['std_bad'] / 2
                    filtered_df['layer_depth'] = filtered_df['layer_depth_good']


                else:
                    filtered_df = df[(df['train_partition'] == train_partition) & 
                                      (df['test_partition'] == test_partition) &
                                      (df['correct'] == corr)]

                # if it matches a number, apply mapping
                if train_partition.isdigit():
                    train_type = ID2LANG[int(train_partition)] 
                else:
                    train_type = train_partition

                if args.apply_lang_scaling:
                    scale = 1 /  LANG_SCAL.get(train_type, 1.0)
                    print(f"Applying language scaling for {train_type} with scale {scale}")
                    filtered_df['sum'] = filtered_df['sum'] * scale
                    filtered_df['mean'] = filtered_df['mean'] * scale
                    filtered_df['std'] = filtered_df['std'] * scale
                    
                print(len(filtered_df), "rows")
                plot_dirname = os.path.join(plotdir, f"param/diffs={args.diffs}/train={train_type}/test={test_partition}/timeline_{train_type}_{test_partition}")

                if not args.grouping == "none":
                    plot_dirname += f"_grouped_by_{args.grouping}"

                if args.normalized:
                    grouped_sums = filtered_df.groupby('step').agg(step_sum=('sum', lambda x: np.abs(x).sum())).reset_index()
                    filtered_df = filtered_df.merge(grouped_sums, on='step')
                    filtered_df['sum'] = filtered_df['sum'] / (filtered_df['step_sum'] + 1e-10)
                    plot_dirname += "_normalized"
                    print(grouped_sums.head())
                    print(filtered_df.head())

                if not os.path.exists(plot_dirname):
                    os.makedirs(plot_dirname)
                plot = make_timeline(filtered_df, plot_dirname, "sum", test_partition, train_partition, group_by=args.grouping)
                store_figure(plot, plot_dirname)

                # also plot mean and std
                if plot_stats:
                    mean_plot = make_timeline(filtered_df, plot_dirname, "mean", test_partition, train_partition)
                    store_figure(mean_plot, plot_dirname + "_mean")
                    std_plot = make_timeline(filtered_df, plot_dirname, "std", test_partition, train_partition)
                    store_figure(std_plot, plot_dirname + "_std")

                # also make logarithmic version
                # filtered_df['sum'] = filtered_df['sum'].apply(lambda x: np.log(np.abs(x) + 1e-10) * np.sign(x))
                # log_plot_dirname = os.path.join(plotdir, f"timeline_{train_partition}_{test_partition}_log/plot")
                # if not os.path.exists(log_plot_dirname):
                #     os.makedirs(log_plot_dirname)
                # log_plot = make_timeline(filtered_df, log_plot_dirname, "sum", test_partition, train_partition)
                # store_figure(log_plot, log_plot_dirname)

                # same for layer_df
                # filtered_layer_df = layer_df[(layer_df['train_partition'] == train_partition) & 
                #                             (layer_df['test_partition'] == test_partition) &
                #                             (layer_df['correct'] == corr)]
                # print(len(filtered_layer_df), "rows")
                # layer_plot_dirname = os.path.join(plotdir, f"{corr}/block/timeline_by_block_{train_partition}_{test_partition}/plot")
                # if not os.path.exists(layer_plot_dirname):
                #     os.makedirs(layer_plot_dirname)
                # layer_plot = make_timeline(filtered_layer_df, layer_plot_dirname, "sum", test_partition, train_partition, layer_order=BLOCK_ORDER)
                # store_figure(layer_plot, layer_plot_dirname)
            


if __name__ == "__main__":
    checkpoint = main()




   