import pandas as pd
from scipy.stats import f_oneway

class factor_select:
    def __init__(self, data: pd.DataFrame):
        """
        初始化函数，保留基础设置与因子选择功能。

        Args:
            data (pd.DataFrame): 数据框, 索引包含时间信息(DatetimeIndex)。
            interval (str): 时间间隔（如'M'表示按月分组）。
        """
        # 基础列：保留日期和收益率等
        self.base_columns = ['permno', 'stock_exret']

        # 因子列名（假设因子从第2列开始）
        self.selected_factors = data.iloc[:, 2:].columns

    def rankIC_method(self, data: pd.DataFrame, num: int) -> pd.DataFrame:
        """
        从历史数据中筛选出与收益率（stock_exret）Rank IC（秩相关系数）绝对值最高的前num个因子。

        Args:
            num: 保留因子数
        Returns:
            pd.DataFrame: 筛选后的数据集，仅保留最相关的num个因子
        """
        # 创建数据副本用于标准化处理，避免修改原始数据
        standardized_data = data.copy()

        # 删除 stock_exret 中的缺失值
        standardized_data.dropna(subset=['stock_exret'], inplace=True)

        # 填充因子列的缺失值为分组的中位数
        standardized_data = standardized_data.groupby(level=0).transform(lambda x: x.fillna(x.median()))

        # 确定需要标准化的列
        no_z = ['permno', 'stock_exret']
        cols_to_standardize = list(set(self.selected_factors) - set(no_z))

        # 定义按月标准化函数
        def zscore_standardize(group):
            return (group - group.mean()) / group.std(ddof=0)

        # 对需要标准化的列按月进行标准化
        standardized_data[cols_to_standardize] = standardized_data.groupby(level=0)[cols_to_standardize].transform(zscore_standardize)

        # 计算每个因子与收益率的 Rank IC
        from scipy.stats import spearmanr
        corr_dict = {}
        for factor in self.selected_factors:
            corr, _ = spearmanr(standardized_data['stock_exret'], standardized_data[factor])
            corr_dict[factor] = corr

        # 按照相关系数的绝对值排序，选择前num个最相关的因子，相关系数至少为0.008
        sorted_factors = sorted(corr_dict.items(), key=lambda x: abs(x[1]), reverse=True)[:num]
        sorted_factors = [factor for factor in sorted_factors if abs(factor[1]) > 0.008]

        # 获取这些因子的名称
        self.selected_factors = [factor for factor, _ in sorted_factors]

        # 保留这些因子以及基础列的数据
        selected_columns = self.base_columns + self.selected_factors
        selected_data = data[selected_columns]  # 原始数据中筛选出对应的列

        return selected_data

    def ANOVA_method(self, data: pd.DataFrame, num: int) -> pd.DataFrame:
        """
        从历史数据中筛选出按大小分组后收益率（stock_exret）差异最大的前 num 个因子。

        Args:
            data: 原始数据集。
            num: 要保留的因子数量。

        Returns:
            pd.DataFrame: 筛选后的数据集，仅保留差异最大的 num 个因子。
        """
        # 按时间间隔对数据进行分组并生成排名
        ranked_data = data.copy()

        for factor in self.selected_factors:
            ranked_data[factor] = (ranked_data[factor].rank(method='first') - 1) // (ranked_data[factor].size // 10 + 1)  # 按特征值排序生成排名

        # 存储 ANOVA 结果
        from scipy.stats import f_oneway
        anova_results = {}

        # 遍历每个因子，计算分组收益率的 ANOVA
        for factor in self.selected_factors:
            f_values = []

            grouped = ranked_data.groupby(ranked_data[factor])  # 按因子排名分组

            # 按分组计算收益率
            for _, group in grouped:
                f_values.append(group['stock_exret'].values)

            # 如果分组数不足以计算 ANOVA，跳过
            if len(f_values) > 1:
                f_statistic = f_oneway(*f_values)[0]  # ANOVA F 检验
                anova_results[factor] = f_statistic  # 记录 F 统计量

        # 按 F 统计量排序，选择前 num 个因子
        sorted_factors = sorted(anova_results.items(), key=lambda x: x[1], reverse=True)[:num]
        self.selected_factors = [factor for factor, _ in sorted_factors]

        # 保留这些因子以及基础列的数据
        selected_columns = self.base_columns + self.selected_factors
        selected_data = data[selected_columns]

        return selected_data