    # 设置画布大小和分辨率
    plt.figure(figsize=(10, 6), dpi=100)
    x_steps = np.arange(len(y_cost))
    y_cost = np.array(y_cost)
    # 绘制核心曲线：指定颜色、线型、标记、标签
    plt.plot(
    x_steps, y_cost, 
    color='#2E86AB',  # 自定义蓝色（美观且专业）
    linestyle='-',    # 实线
    marker='.',       # 每个点标记为小圆点
    markersize=4,     # 标记大小
    label='Objective Function (y_cost)'
    )

    # -------------------------- 3. 图表美化与标注 --------------------------
    # 设置标题和坐标轴标签（支持中文，需指定字体）
    plt.title('Trend of Objective Function (y_cost) During Optimization', fontsize=14, fontweight='bold')
    plt.xlabel('Iteration Step', fontsize=12)
    plt.ylabel('y_cost Value', fontsize=12)

    # 添加网格（便于读取数值）
    plt.grid(True, linestyle='--', alpha=0.6)

    # 添加图例
    plt.legend(loc='upper right', fontsize=10)

    # 可选：添加最小值标注（优化场景常用）
    min_idx = np.argmin(y_cost)
    min_step = x_steps[min_idx]
    min_cost = y_cost[min_idx]
    plt.annotate(
    f'Min: {min_cost:.2f}\nStep: {min_step}',  # 标注文本
    xy=(min_step, min_cost),                  # 标注目标点
    xytext=(min_step + 5, min_cost + 5),      # 文本位置
    arrowprops=dict(arrowstyle='->', color='red'),  # 箭头
    fontsize=10,
    color='red',
    fontweight='bold'
    )

    # -------------------------- 4. 显示/保存图表 --------------------------
    # 显示图表
    plt.tight_layout()  # 自动调整布局，避免标签被截断
    plt.show()

    可选：保存图表（高清格式，可用于论文/报告）
    plt.savefig('y_cost_trend.png', dpi=300, bbox_inches='tight')