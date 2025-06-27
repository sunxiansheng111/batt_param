
import streamlit as st
import numpy as np
import pandas as pd
from scipy import signal
import matplotlib.pyplot as plt
from sklearn.metrics import root_mean_squared_error, mean_absolute_error
from pybatteryid import ModelStructure
from pybatteryid.identification import identify_model
from pybatteryid.utilities import print_model_details, analyze_dataset
from pybatteryid.plotter import plot_time_vs_current, plot_time_vs_voltage
from pybatteryid.simulation import simulate_model

# 设置中文字体
plt.rcParams["font.family"] = ["SimHei", "WenQuanYi Micro Hei", "Heiti TC"]
plt.rcParams["axes.unicode_minus"] = False  # 解决负号显示问题

# 页面设置
st.set_page_config(
    page_title="电池模型辨识与验证",
    page_icon="🔋",
    layout="wide"
)

# 标题
st.title("电池模型辨识与验证系统")
st.markdown("该应用程序允许您通过界面设置参数并执行电池模型的辨识和验证过程。")

# 侧边栏 - 参数设置
with st.sidebar:
    st.header("模型参数设置")

    # 上传数据文件
    uploaded_file = st.file_uploader("上传数据文件", type=["xlsx", "csv"])

    # 电池容量设置
    battery_capacity = st.number_input(
        "电池容量 (mAh)",
        min_value=0.0,
        value=307.0,
        step=1.0,
        help="电池的额定容量，单位为毫安时(mAh)"
    )
    battery_capacity = battery_capacity * 3600  # 转换为As

    # 采样周期设置
    sampling_period = st.number_input(
        "采样周期 (秒)",
        min_value=0.01,
        value=1.0,
        step=0.1,
        help="数据采集的时间间隔，单位为秒"
    )

    # 初始SOC设置
    initial_soc = st.slider(
        "初始SOC",
        min_value=0.0,
        max_value=1.0,
        value=1.0,
        step=0.05,
        help="电池的初始荷电状态"
    )

    # SOC-OCV查找表设置
    st.subheader("SOC-OCV查找表")
    st.write("设置SOC与OCV的对应关系（数组格式）")

    # 默认SOC值
    default_soc = [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5,
                   0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0]

    # 默认OCV值
    default_ocv = [2.8133, 3.1628, 3.2039, 3.2239, 3.2482, 3.2648, 3.2832,
                   3.2913, 3.2922, 3.2926, 3.293, 3.2948, 3.2997, 3.3305,
                   3.3315, 3.3317, 3.332, 3.3322, 3.3325, 3.3333, 3.3626]

    # 将默认值转换为字符串格式
    default_soc_str = ", ".join([str(x) for x in default_soc])
    default_ocv_str = ", ".join([str(x) for x in default_ocv])

    # 使用文本区域输入SOC和OCV数组
    soc_input = st.text_area(
        "SOC 值 (逗号分隔)",
        value=default_soc_str,
        height=100,
        help="输入一系列0到1之间的SOC值，用逗号分隔"
    )

    ocv_input = st.text_area(
        "OCV 值 (V，逗号分隔)",
        value=default_ocv_str,
        height=100,
        help="输入对应的OCV值（伏特），用逗号分隔"
    )

    # 解析输入的数组
    try:
        soc_values = [float(x.strip()) for x in soc_input.split(',') if x.strip()]
        ocv_values = [float(x.strip()) for x in ocv_input.split(',') if x.strip()]

        # 检查输入是否有效
        if len(soc_values) < 2:
            st.warning("SOC值数量至少需要2个")
        elif len(soc_values) != len(ocv_values):
            st.warning("SOC和OCV值的数量必须相同")
        else:
            # 检查SOC值是否在0-1范围内且递增
            if any(x < 0 or x > 1 for x in soc_values):
                st.warning("SOC值必须在0到1之间")
            if any(soc_values[i] > soc_values[i + 1] for i in range(len(soc_values) - 1)):
                st.warning("SOC值必须按升序排列")

    except ValueError:
        st.error("无法解析输入值，请确保输入的是有效的数字，并用逗号分隔")
        soc_values = default_soc
        ocv_values = default_ocv

    # 模型结构设置
    st.subheader("模型结构设置")
    model_order = st.slider("模型阶数", min_value=1, max_value=10, value=2, step=1)
    nonlinearity_order = st.slider("非线性阶数", min_value=1, max_value=5, value=2, step=1)

    # 基函数初始化 - 自动包含基础函数，不显示UI
    basis_functions = ['1/s', 's']  # 自动添加基础函数

    # 分数阶导数设置
    st.subheader("分数阶导数设置")
    use_fractional_derivative = st.checkbox("使用分数阶导数", value=True)

    if use_fractional_derivative:
        # 创建一个会话状态来保存分数阶导数的值
        if 'alpha' not in st.session_state:
            st.session_state.alpha = 0.5

        # 使用滑块控制第一个参数，第二个参数自动计算
        alpha = st.slider(
            "分数阶导数参数 α",
            min_value=0.1,
            max_value=0.9,
            value=st.session_state.alpha,
            step=0.1,
            help="设置分数阶导数的第一个参数，第二个参数将自动设为 1-α"
        )

        # 更新会话状态
        st.session_state.alpha = alpha

        # 计算第二个参数
        beta = 1.0 - alpha

        # 显示计算出的第二个参数
        st.text(f"分数阶导数参数 β: {beta:.1f} (自动计算为 1-α)")

        # 生成基函数表示
        fractional_basis_function = f"d[{alpha:.1f},{beta:.1f}]"
        basis_functions.append(fractional_basis_function)
        st.info(f"已添加分数阶导数基函数: {fractional_basis_function}")



    # 运行按钮
    run_analysis = st.button("运行分析", key="run_button")

# 主内容区
if run_analysis:
    if uploaded_file is None:
        st.error("请上传数据文件")
    else:
        # 显示进度条
        progress_bar = st.progress(0)
        status_text = st.empty()

        try:
            # 1. 加载数据
            status_text.text("正在加载数据...")
            if uploaded_file.name.endswith('.xlsx'):
                dataset = pd.read_excel(uploaded_file)
            else:
                dataset = pd.read_csv(uploaded_file)

            # 检查必要的列
            required_columns = ['time', 'current', 'voltage']
            missing_columns = [col for col in required_columns if col not in dataset.columns]
            if missing_columns:
                raise ValueError(f"数据文件缺少必要的列: {', '.join(missing_columns)}")

            progress_bar.progress(10)
            status_text.text("数据加载完成，正在预处理...")


            # 2. 数据预处理函数
            def preprocess_battery_data(dataset, battery_capacity, initial_soc):
                """预处理电池数据"""
                st.write("=== 数据预处理开始 ===")

                # 1. 检查和处理无穷大值和NaN
                for col in ['time', 'current', 'voltage']:
                    if col in dataset.columns:
                        # 替换无穷大值为NaN
                        dataset[col] = dataset[col].replace([np.inf, -np.inf], np.nan)

                        # 检查NaN值
                        nan_count = dataset[col].isna().sum()
                        if nan_count > 0:
                            st.warning(f"{col} 列发现 {nan_count} 个NaN值，将进行插值处理")
                            # 使用线性插值填充NaN
                            dataset[col] = dataset[col].interpolate(method='linear')
                            # 如果首尾有NaN，使用前向/后向填充
                            dataset[col] = dataset[col].fillna(method='ffill').fillna(method='bfill')

                # 2. 数据平滑处理
                if len(dataset) > 10:
                    window_length = min(5, len(dataset) // 10)
                    if window_length >= 3 and window_length % 2 == 0:
                        window_length += 1  # 确保是奇数

                    if window_length >= 3:
                        dataset['current'] = signal.savgol_filter(dataset['current'], window_length, 1)
                        st.write(f"已对电流数据应用Savitzky-Golay滤波，窗口长度: {window_length}")

                # 5. 规范化时间序列
                time_values = dataset['time'].values
                time_values = time_values - time_values[0]  # 从0开始

                # 6. 准备最终数据集
                processed_dataset = {
                    'initial_soc': initial_soc,
                    'time_values': time_values,
                    'current_values': dataset['current'].values,
                    'voltage_values': dataset['voltage'].values
                }

                st.write("=== 数据预处理完成 ===\n")
                return processed_dataset


            # 3. 预处理数据
            identification_dataset = preprocess_battery_data(dataset, battery_capacity, initial_soc)

            progress_bar.progress(30)
            status_text.text("数据预处理完成，正在初始化模型结构...")

            # 4. 初始化模型结构
            model_structure = ModelStructure(
                battery_capacity=battery_capacity,
                sampling_period=sampling_period
            )

            # 设置EMF函数
            model_structure.add_emf_function(
                {'soc_values': np.array(soc_values), 'voltage_values': np.array(ocv_values)})

            # 可视化SOC-OCV曲线
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.plot(soc_values, ocv_values, 'o-')
            ax.set_xlabel('SOC')
            ax.set_ylabel('OCV (V)')
            ax.set_title('SOC-OCV曲线')
            ax.grid(True)
            st.pyplot(fig)

            # 显示SOC-OCV表格
            st.subheader("SOC-OCV 数据点")
            soc_ocv_df = pd.DataFrame({
                'SOC': soc_values,
                'OCV (V)': ocv_values
            })
            st.dataframe(soc_ocv_df)

            progress_bar.progress(40)
            status_text.text("模型结构初始化完成，正在分析数据集...")

            # 5. 分析数据集
            analyze_dataset(identification_dataset, battery_capacity, sampling_period, model_structure.emf_function)

            progress_bar.progress(50)
            status_text.text("数据集分析完成，正在辨识模型...")

            # 6. 模型识别
            try:
                # 设置基函数
                model_structure.basis_functions = []
                model_structure.add_basis_functions(basis_functions)

                # 尝试识别模型
                model = identify_model(
                    identification_dataset,
                    model_structure,
                    model_order=model_order,
                    nonlinearity_order=nonlinearity_order,
                    optimizers=['ridgecv.sklearn']  # 使用更稳定的优化器
                )

                st.success(
                    f"模型识别成功！\n- 模型阶数: {model_order}\n- 非线性阶数: {nonlinearity_order}\n- 基函数: {', '.join(basis_functions)}")
                progress_bar.progress(70)
                status_text.text("模型辨识完成，正在验证模型...")


                # 7. 模型验证
                def validate_model(model, dataset, model_structure, battery_capacity, sampling_period):
                    """验证模型并绘制电压对比曲线"""
                    st.write("\n=== 开始模型验证，请等待 ===")

                    no_of_initial_values = 4
                    if len(dataset['voltage_values']) < no_of_initial_values:
                        no_of_initial_values = len(dataset['voltage_values'])

                    current_profile = {
                        'initial_soc': initial_soc,
                        'time_values': dataset['time_values'],
                        'current_values': dataset['current_values'],
                        'voltage_values': dataset['voltage_values'][:no_of_initial_values]
                    }

                    # 模型仿真
                    voltage_simulated = simulate_model(model, current_profile)

                    # 绘制实际电压与仿真电压对比曲线
                    fig, ax = plt.subplots(figsize=(12, 8))
                    ax.plot(dataset['time_values'] / 3600, dataset['voltage_values'], 'k-', label='measure_voltage')
                    ax.plot(dataset['time_values'] / 3600, voltage_simulated, color='orangered', linestyle='--', label='model_voltage')
                    ax.set_xlabel('时间 (h)')
                    ax.set_ylabel('电压 (V)')
                    ax.set_title('measure_voltage and model_voltage')
                    ax.legend()
                    ax.grid(True)
                    st.pyplot(fig)

                    # 误差分析
                    rmse = root_mean_squared_error(
                        voltage_simulated[no_of_initial_values:],
                        dataset['voltage_values'][no_of_initial_values:]
                    )
                    mae = mean_absolute_error(
                        voltage_simulated[no_of_initial_values:],
                        dataset['voltage_values'][no_of_initial_values:]
                    )

                    st.write(f"\n=== 误差分析结果 ===")
                    st.write(f"RMSE: {rmse * 1000:.2f} mV")
                    st.write(f"MAE: {mae * 1000:.2f} mV")

                    # 绘制误差曲线
                    fig, ax = plt.subplots(figsize=(12, 6))
                    error = voltage_simulated[no_of_initial_values:] - dataset['voltage_values'][no_of_initial_values:]
                    ax.plot(dataset['time_values'][no_of_initial_values:] / 3600, error, 'b-')
                    ax.axhline(y=0, color='r', linestyle='--')
                    ax.set_xlabel('时间 (h)')
                    ax.set_ylabel('电压误差 (V)')
                    ax.set_title('电压误差曲线')
                    ax.grid(True)
                    st.pyplot(fig)

                    return voltage_simulated


                # 验证模型
                validate_model(model, identification_dataset, model_structure, battery_capacity, sampling_period)

                progress_bar.progress(100)
                status_text.text("分析完成！")
                st.balloons()

            except Exception as e:
                st.error(f"模型识别失败: {str(e)}")
                import traceback

                st.write(traceback.format_exc())

        except Exception as e:
            st.error(f"程序执行出错: {str(e)}")
            import traceback

            st.write(traceback.format_exc())
