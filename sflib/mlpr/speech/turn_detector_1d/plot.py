# デモ用のプロット
from .base import TurnDetector1d
from ....corpus.speech.duration_v2 import DurationInfoV2
import plotly.graph_objects as go
import numpy as np

# この import は暫定（内容が変わるかも知れないが現状はこれでOK）
from .trainer0013 import extract_ut_st_zt_ft_from_duration_info


def predict(turn_detector: TurnDetector1d,
            wav, st, frame_rate):
    """TurnDetector1d で各種予測を行う．
    """
    sub_wav_len = 32000
    sub_st_len = int(sub_wav_len / 16000 * frame_rate)
    total_sub = len(wav) // sub_wav_len + 1

    sub_wav_idx = 0
    sub_st_idx = 0
    result_list = []
    turn_detector.reset()
    turn_detector.torch_model.eval()
    while sub_wav_idx < len(wav):
        wav_start = sub_wav_idx * sub_wav_len
        wav_end = wav_start + sub_wav_len
        wav_sub = wav[wav_start:wav_end]
        st_start = sub_st_idx * sub_st_len
        st_end = st_start + sub_st_len
        st_sub = st[st_start:st_end]
        if len(wav_sub) <= 0:
            break
        result = turn_detector.predict([wav_sub], [st_sub])
        if result is None:
            break
        result = result[0][:, 0, :]
        result_list.append(result)
        sub_wav_idx += 1
        sub_st_idx += 1
        print('\r{}/{}'.format(sub_wav_idx, total_sub), end='', flush=True)
    return np.concatenate(result_list, axis=0)
        

def generate_datetime_x(length, frame_rate):
    """時系列データにい対するX（時間軸）の値を生成する．
    """
    import datetime
    x = np.arange(length) / frame_rate
    zero_day = datetime.datetime(1, 1, 1)
    x = [zero_day + datetime.timedelta(seconds=s) for s in x]
    return x


def generate_impulse_series(values, frame_rate,
                            threshold=0.9,
                            y_value=1.0,
                            marker_size=10):
    """時系列データに対してインパルス状のプロットをするための準備をする．
    
    Returns:
      x(list): X座標のリスト
      y(list): Y座標のリスト
      s(list): Sマーカーサイズのリスト
    """
    raw_x = generate_datetime_x(len(values), frame_rate)
    indices = np.where(values > threshold)
    indices = indices[0]
    x = []
    for idx in indices:
        x.extend([raw_x[idx]] * 4)
    y = [0, y_value, 0, None] * len(indices)
    s = [0, marker_size, 0, 0] * len(indices)
    return x, y, s


def plot(turn_detector: TurnDetector1d,
         duration_info: DurationInfoV2, ch: int):
    ch_ut = ch
    ch_st = 1 - ch
    frame_rate = turn_detector.input_calculator.feature_rate
    
    labels = extract_ut_st_zt_ft_from_duration_info(
        duration_info, ch_ut, ch_st, frame_rate)

    st = labels[:, 1]
    wav = duration_info.wav[ch_ut]
    result = predict(turn_detector, wav, st, frame_rate)

    if labels.shape[0] < result.shape[0]:
        result = result[:labels.shape[0]]
    elif labels.shape[0] > result.shape[0]:
        labels = labels[:result.shape[0]]
    ut = labels[:, 0]
    st = labels[:, 1]
    zt = labels[:, 2]
    ft = labels[:, 3]
    yt_pred = result[:, 0]
    at_locked = result[:, 1]
    at_unlocked = result[:, 2]
    if result.shape[1] <= 4:
        ut_pred = result[:, 3]
    else:
        ut_pred = result[:, 4]

    # yt_pred が 0.8 を超えるタイミング（z(t)の予測値）を計算
    yt1_pred = np.concatenate([[0.0], yt_pred[:-1]])
    zt_pred = np.float32((yt1_pred <= 0.8) & (yt_pred > 0.8))
    
    # X座標は全部の系列で共通
    raw_x = generate_datetime_x(len(ut), frame_rate)
    
    # ztとzt_predはインパルス状の表示
    x_zt, y_zt, s_zt = generate_impulse_series(zt, frame_rate)
    x_zt_pred, y_zt_pred, s_zt_pred = generate_impulse_series(zt_pred, frame_rate)

    # --- 準備は整ったのでプロット ---
    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=raw_x, y=ut,
            name='u(t) (true)',
            mode='lines',
            yaxis='y',
            line=dict(
                width=1.5,
                color='dodgerblue',
            ),
        )
    )

    fig.add_trace(
        go.Scatter(
            x=raw_x, y=ut_pred,
            name='u(t) (estimated)',
            mode='lines',
            fill='tozeroy',
            yaxis='y',
            line=dict(
                width=0,
                color='dodgerblue',
            ),
        )
    )

    fig.add_trace(
        go.Scatter(
            x=x_zt, y=y_zt,
            name='z(t) (true)',
            mode='lines+markers',
            yaxis='y',
            marker=dict(
                size=s_zt,
                color='red',
            ),
            line=dict(
                color='red',
            ),
        )
    )

    fig.add_trace(
        go.Scatter(
            x=raw_x, y=yt_pred,
            name='y(t) (estimated)',
            mode='lines',
            yaxis='y',
            line=dict(
                color='maroon',
                width=2,
            ),
        )
    )

    fig.add_trace(
        go.Scatter(
            x=x_zt_pred, y=y_zt_pred,
            name='z(t) (estimated)',
            mode='lines+markers',
            yaxis='y',
            marker=dict(
                size=s_zt_pred,
                color='brown',
                symbol='x',
            ),
            line=dict(
                color='brown',
            ),
        )
    )

    fig.add_trace(
        go.Scatter(
            x=[raw_x[0], raw_x[-1]],
            y=[0.8, 0.8],
            name='',
            mode='lines',
            yaxis='y',
            line=dict(
                width=1,
                color='black',
                dash='dot',
            ),
            showlegend=False,
        )
    )

    fig.add_trace(
        go.Scatter(
            x=raw_x, y=at_unlocked,
            name='alpha(t)',
            mode='lines',
            line=dict(
                width=1,
                color='red',
                dash='dot',
            ),
            yaxis='y2',
        )
    )
    
    fig.add_trace(
        go.Scatter(
            x=raw_x, y=at_locked,
            name='alpha(t) (locked)',
            mode='lines',
            line=dict(
                width=1.5,
                color='red',
            ),
            yaxis='y2',
        )
    )

    fig.add_trace(
        go.Scatter(
            x=raw_x, y=st,
            name='s(t) (true)',
            mode='lines',
            fill='tozeroy',
            yaxis='y3',
            line=dict(
                width=0,
                color='red',
            ),
            showlegend=False,
        )
    )

    fig.add_trace(
        go.Scatter(
            x=raw_x, y=ft,
            name='f(t)',
            mode='lines',
            fill='tozeroy',
            yaxis='y3',
            line=dict(
                width=0,
                color='blue',
            ),
            showlegend=False,
        )
    )

    fig.update_layout(
        yaxis=dict(
            # anchor="x",
            autorange=False,
            domain=[0.4, 1.0],
            linecolor="#673ab7",
            mirror=True,
            range=[0, 1.1],
            showline=True,
            side="right",
            tickfont={"color": "#673ab7"},
            tickmode="array",
            tickvals=[0.0, 0.5, 0.8, 1.0],
            titlefont={"color": "#673ab7"},
            type="linear",
            zeroline=True,
            zerolinecolor='black',
        ),
        yaxis2=dict(
            # anchor="x",
            autorange=True,
            domain=[0, 0.3],
            linecolor="#E91E63",
            mirror=True,
            range=[0.0, 0.05],
            showline=True,
            side="right",
            tickfont={"color": "#E91E63"},
            tickmode="auto",
            ticks="",
            titlefont={"color": "#E91E63"},
            type="linear",
            zeroline=True,
            zerolinecolor='black',
        ),
        yaxis3=dict(
            # anchor="x",
            autorange=True,
            domain=[0.3, 0.4],
            linecolor="#E91E63",
            mirror=True,
            range=[0.0, 1.0],
            showline=True,
            side="right",
            tickfont={"color": "#E91E63"},
            tickmode="array",
            tickvals=[],
            titlefont={"color": "#E91E63"},
            type="linear",
            zeroline=False,
            zerolinecolor='black',
        ),
        xaxis=dict(
            anchor='y2',
            tickformat="%H:%M:%S.%2f",
            # rangemode="tozero",
            range=[raw_x[0], raw_x[int(frame_rate * 10)]],
            rangeslider=dict(
                visible=True
            ),
            type="date"
        ),
        height=500,
        legend=dict(borderwidth=1, x=1.1),
        dragmode="pan",
    )

    # def ppp(trace, points, selector):
    #     print(points)
    # fig.data[3].on_click(ppp)
    # fig.show()
    return fig, dict(ut_pred=ut_pred,
                     yt_pred=yt_pred,
                     zt=zt,
                     at_unlocked=at_unlocked,
                     at_locked=at_locked,
                     datetime_x=raw_x)


