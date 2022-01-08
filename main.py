import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from sklearn import linear_model
from pyecharts import options as opts
from pyecharts.charts import Map


class DataAnalysis:
    def __init__(self):
        self.newCases = pd.read_csv("newCases.csv")
        self.totalCases = pd.read_csv("totalCases.csv")
        self.totalDeaths = pd.read_csv("totalDeaths.csv")
        self.peopleData = pd.read_csv("population.csv")
        self.vaccine = pd.read_csv("vaccine.csv")
        self.unitedNouns = ["World", "High income", "Upper middle income", "Asia", "Europe", "Lower middle income",
                            "North America", "European Union", "Africa", "South America"]

    # 15天中，全球新冠疫情的总体变化趋势
    def worldTrend(self):
        df = self.totalCases
        df = df[df["country"] == "World"]  # 筛选出世界所有确诊病例
        # 依次取出确诊病例、对应日期
        casesList, dateList = [df.loc[i][1] for i in range(0, 15)], [df.loc[i][2] for i in range(0, 15)]
        # 实例化图片、坐标轴
        fig, ax = plt.subplots(figsize=(12, 10))
        # 设置纵坐标范围
        delta = (max(casesList) - min(casesList)) / 20
        ax.set_ylim(min(casesList) - delta, max(casesList) + delta)
        # 设置图片标题、横坐标标题、纵坐标标题
        plt.title('15天内世界疫情整体趋势')
        plt.xlabel('时间', fontsize=9)
        plt.ylabel('确诊人数', fontsize=9)
        # 旋转横坐标，防止重叠
        plt.xticks(rotation=45, fontsize=9)
        # 设置边框部分留白
        plt.gcf().subplots_adjust(bottom=0.2, left=0.2)
        plt.tight_layout()
        # 填充数据，设置颜色
        plt.bar(dateList[::-1], casesList[::-1], color='lightcoral')  # bar的颜色改为红色
        for a, b in zip(dateList, casesList):  # 在直方图上显示本学期的排名数字
            plt.text(a, b + 0.2, '%d' % b, ha='center', va='bottom', fontsize=10)
        # 展示图片
        plt.show()

    # 15天中，每日新增确诊数累计排名前 10 个国家的每日新增确诊数据的曲线图
    def newCasesTop10(self):
        df = self.newCases
        # 按国家groupby，求和并排序
        newDf = {'country': [], 'newcasesSum': []}
        # 求和，构造df
        for eachCountry in df.groupby('country'):
            if eachCountry[0] in self.unitedNouns:
                continue
            newDf['country'].append(eachCountry[0])
            newDf['newcasesSum'].append(eachCountry[1]['newCases'].sum())
        newDf = pd.DataFrame(newDf).sort_values(by='newcasesSum', ascending=False)
        # 获取新增排名前10的国家
        countries = [i for i in newDf['country'][:10]]
        dataDic = {}
        for country in countries:
            countryDf = df[df['country'] == country]
            dataDic[country] = [cases for cases in countryDf['newCases']]
        dateList = ['2021/12/{}'.format(i) for i in range(15, 30)]
        # 实例化图片、坐标轴
        plt.subplots(figsize=(12, 10))
        for country in countries:
            plt.plot(dateList, dataDic[country][::-1], label=country)
        # 旋转横坐标，防止重叠
        plt.xticks(rotation=45, fontsize=9)
        plt.title('15天内新增总数排前10名的国家新增病例趋势')
        plt.xlabel('时间', fontsize=9)
        plt.ylabel('新增人数', fontsize=9)
        plt.legend()  # 显示图例
        plt.show()

    # 确诊总数排前10的国家
    def totalCasesTop10(self):
        df = self.totalCases
        newDf = df[df['date'] == '2021/12/29'].sort_values(by='totalCases', ascending=False)
        # 获取新增排名前10的国家
        countries, totalcasesSum = [], []
        for row in newDf.iterrows():
            if row[1][0] not in self.unitedNouns:
                countries.append(row[1][0])
                totalcasesSum.append(row[1][1])
            if len(countries) == 10:
                break
        plt.subplots(figsize=(12, 10))
        plt.title('确诊人数Top10')
        plt.xlabel('国家', fontsize=9)
        plt.ylabel('确诊人数', fontsize=9)
        # 旋转横坐标，防止重叠
        plt.xticks(rotation=45, fontsize=9)
        plt.tight_layout()
        # 填充数据，设置颜色
        plt.bar(countries, totalcasesSum, color='coral')  # bar的颜色改为红色
        # 设置边框部分留白
        plt.gcf().subplots_adjust(bottom=0.1)
        # 展示图片
        for a, b in zip(countries, totalcasesSum):  # 在直方图上显示本学期的排名数字
            plt.text(a, b + 0.2, '%d' % b, ha='center', va='bottom', fontsize=10)
        plt.show()

    # 各国占确诊人数比例
    def rateOfEachCountry(self):
        df = self.totalCases
        df = df[df['country'] == 'World']  # 筛选出世界所有确诊病例
        df = df[df['date'] == '2021/12/29']
        worldPopulation = df['totalCases'][0]
        df = self.totalCases
        newDf = df[df['date'] == '2021/12/29'].sort_values(by='totalCases', ascending=False)
        # 获取百分比占10排名前的国家
        countries, rates = [], []
        curSum = 0
        for row in newDf.iterrows():
            if row[1][0] not in self.unitedNouns:
                countries.append(row[1][0])
                curSum += row[1][1]
                rate = row[1][1] * 100 / worldPopulation
                rates.append(float('%.3f' % rate))
            if curSum / worldPopulation > 0.78:
                break
        countries.append('Others')
        newRate = 100 - sum(rates)
        rates.append(newRate)
        explode = [0.3, 0.2, 0.1] + [0] * (len(rates) - 3)
        plt.subplots(figsize=(10, 8))
        plt.title('各国确诊病例占全球确诊病例比例图')
        plt.pie(rates, labels=countries, explode=explode, labeldistance=1.2, autopct='%1.1f%%', shadow=False,
                startangle=90,
                pctdistance=0.6)
        plt.show()

    # 感染率
    def rateTop10(self):
        casedf, populationdf = self.totalCases.dropna(), self.peopleData.dropna()
        newDf = casedf[casedf['date'] == '2021/12/29']
        result = pd.merge(newDf, populationdf, on='country')
        result['population'] = result['population'].map(lambda x: float(x))
        result['rate'] = result['totalCases'] / result['population']
        result = result.sort_values(by='rate', ascending=False)
        countries, rates = [], []
        for row in result.iterrows():
            if row[1][0] not in self.unitedNouns:
                countries.append(row[1][0])
                rates.append(float('%.2f' % (100 * float(row[1][4]))))
            if len(countries) == 10:
                break
        fig, ax = plt.subplots(figsize=(12, 8))
        # 设置纵坐标范围
        delta = (max(rates) - min(rates)) / 20
        ax.set_ylim(min(rates) - delta, max(rates) + delta)
        # 设置图片标题、横坐标标题、纵坐标标题
        plt.title('确诊比例top10国家')
        plt.xlabel('国家', fontsize=9)
        plt.ylabel('确诊人数百分比', fontsize=9)
        # 旋转横坐标，防止重叠
        plt.xticks(rotation=45, fontsize=9)
        # 设置边框部分留白
        plt.gcf().subplots_adjust(bottom=0.1)
        # plt.tight_layout()
        # 填充数据，设置颜色
        plt.bar(countries, rates, color='darkorange')  # bar的颜色改为红色
        for a, b in zip(countries, rates):  # 在直方图上显示本学期的排名数字
            plt.text(a, b + 0.2, '%.2f' % b, ha='center', va='bottom', fontsize=10)
        # 展示图片
        plt.show()

    # 疫苗接种人数
    def vaccineNum(self):
        df = self.vaccine
        newDf = df[df['date'] == '2021/12/29'].sort_values(by='vaccine', ascending=False)
        countries, vaccines = [], []
        for row in newDf.iterrows():
            if row[1][0] not in self.unitedNouns:
                countries.append(row[1][0])
                vaccines.append(row[1][1])

        Map().add(
            "疫苗接种人数",
            [list(z) for z in zip(countries, vaccines)],
            is_map_symbol_show=False,
            maptype="world",
            label_opts=opts.LabelOpts(is_show=True),
            itemstyle_opts=opts.ItemStyleOpts(color="rgb(49,60,72)")
        ).set_series_opts(label_opts=opts.LabelOpts(is_show=False)).set_global_opts(
            title_opts=opts.TitleOpts(title="全球疫苗接种情况"),
            visualmap_opts=opts.VisualMapOpts(max_=150000000),
        ).render("vaccineMap.html")

    # 疫苗接种率
    def vaccineRate(self):
        casedf, populationdf = self.vaccine.dropna(), self.peopleData.dropna()
        newDf = casedf[casedf['date'] == '2021/12/29']
        result = pd.merge(newDf, populationdf, on='country')
        result['population'] = result['population'].map(lambda x: float(x))
        result['rate'] = result['vaccine'] / result['population']
        result = result.sort_values(by='rate', ascending=True)
        countries, rates = [], []
        for row in result.iterrows():
            if row[1][0] not in self.unitedNouns:
                countries.append(row[1][0])
                rates.append(float('%.2f' % (100 * float(row[1][4]))))
            if len(countries) == 10:
                break
        fig, ax = plt.subplots(figsize=(12, 8))
        # 设置纵坐标范围
        ax.set_ylim(0, 5)
        # 设置图片标题、横坐标标题、纵坐标标题
        plt.title('疫苗接种率最低的10个国家')
        plt.xlabel('国家', fontsize=9)
        plt.ylabel('疫苗接种人数百分比', fontsize=9)
        # 旋转横坐标，防止重叠
        plt.xticks(rotation=45, fontsize=9)
        # 设置边框部分留白
        plt.gcf().subplots_adjust(bottom=0.15)
        # plt.tight_layout()
        # 填充数据，设置颜色
        plt.bar(countries, rates, color='hotpink')  # bar的颜色改为红色
        for a, b in zip(countries[::-1], rates[::-1]):  # 在直方图上显示本学期的排名数字
            plt.text(a, b + 0.2, '%.2f' % b, ha='center', va='bottom', fontsize=10)
        # 展示图片
        plt.show()

    # GDP排前10的所有国家的总确诊人数，全球 GDP 前十名国家的累计确诊人数箱型图，要有平均值
    def GDPTop10TotalCases(self):
        GDPTop10 = ['United States', 'China', 'Japan', 'Germany', 'India', 'United Kingdom', 'France', 'Italy',
                    'Canada', 'South Korea']
        cases = []
        casedf = self.totalCases.dropna()
        newDf = casedf[casedf['date'] == '2021/12/29']
        for country in GDPTop10:
            tempDf = newDf[newDf['country'] == country]
            cases.append(tempDf['totalCases'].iloc[0])
        # palegreen
        fig, ax = plt.subplots(figsize=(12, 8))
        ax.boxplot(cases, patch_artist=True, showmeans=True)  # 描点上色
        plt.title('全球GDP前10名国家的累计确诊人数箱型图')
        plt.ylabel('确诊人数')
        cases.append((int(sum(cases) / len(cases))))
        for i in cases:
            ax.text(1.2, i, i, verticalalignment='center', fontsize=8, backgroundcolor="white")
        plt.show()  # 展示

    # 死亡率前10
    def deathRateTop10(self):
        deathdf, casesdf = self.totalDeaths.dropna(), self.totalCases.dropna()
        casesDf = casesdf[casesdf['date'] == '2021/12/29']
        deathdf = deathdf[deathdf['date'] == '2021/12/29']
        result = pd.merge(casesDf, deathdf, on='country')
        print(result)
        result['totalCases'] = result['totalCases'].map(lambda x: float(x))
        result['rate'] = result['totalDeaths'] / result['totalCases']
        result = result.sort_values(by='rate', ascending=False)
        countries, rates = [], []
        for row in result.iterrows():
            if row[1][0] not in self.unitedNouns:
                countries.append(row[1][0])
                rates.append(float('%.2f' % (100 * float(row[1][5]))))
            if len(countries) == 10:
                break
        fig, ax = plt.subplots(figsize=(12, 8))
        # 设置纵坐标范围
        delta = (max(rates) - min(rates)) / 20
        ax.set_ylim(min(rates) - delta, max(rates) + delta)
        # 设置图片标题、横坐标标题、纵坐标标题
        plt.title('死亡率top10国家')
        plt.xlabel('国家', fontsize=9)
        plt.ylabel('死亡率', fontsize=9)
        # 旋转横坐标，防止重叠
        plt.xticks(rotation=45, fontsize=9)
        # 设置边框部分留白
        plt.gcf().subplots_adjust(bottom=0.15)
        # 填充数据，设置颜色
        plt.bar(countries, rates, color='gold')  # bar的颜色改为红色
        for a, b in zip(countries, rates):  # 在直方图上显示本学期的排名数字
            plt.text(a, b + 0.2, '%.2f' % b, ha='center', va='bottom', fontsize=10)
        # 展示图片
        plt.show()

    # 应对疫情最好的10个国家
    def bestTop10(self):
        vcdf, casedf, populationdf = self.vaccine.dropna(), self.totalCases.dropna(), self.peopleData.dropna()
        casedf = casedf[casedf['date'] == '2021/12/29']
        vcdf = vcdf[vcdf['date'] == '2021/12/29']
        result = pd.merge(casedf, populationdf, on='country')
        result = pd.merge(result, vcdf, on='country')
        result['population'] = result['population'].map(lambda x: float(x))
        result['casesRate'] = result['totalCases'] / result['population']
        result['vcRate'] = result['vaccine'] / result['population']
        result = result.sort_values(by='casesRate', ascending=True)
        countries, rates = [], []
        for row in result.iterrows():
            if row[1][7] < 0.80 or float(row[1][3]) < 10000000:
                continue
            if row[1][0] not in self.unitedNouns:
                countries.append(row[1][0])
                rates.append(float('%.2f' % (100 * float(row[1][6]))))
            if len(countries) == 10:
                break
        fig, ax = plt.subplots(figsize=(12, 8))
        # 设置纵坐标范围
        delta = (max(rates) - min(rates)) / 20
        ax.set_ylim(min(rates) - delta, max(rates) + delta)
        # 设置图片标题、横坐标标题、纵坐标标题
        plt.title('防疫做的最好的的10个国家')
        plt.xlabel('国家', fontsize=9)
        plt.ylabel('确诊比例', fontsize=9)
        # 旋转横坐标，防止重叠
        plt.xticks(rotation=45, fontsize=9)
        # 设置边框部分留白
        plt.gcf().subplots_adjust(bottom=0.15)
        # 填充数据，设置颜色
        plt.bar(countries, rates, color='paleturquoise')  # bar的颜色改为红色
        for a, b in zip(countries, rates):  # 在直方图上显示本学期的排名数字
            plt.text(a, b + 0.1, '%.2f' % b, ha='center', va='bottom', fontsize=10)
        # 展示图片
        plt.show()

    def predict(self):
        df = self.totalCases
        df = df[df["country"] == "World"]  # 筛选出世界所有确诊病例
        # 依次取出确诊病例、对应日期
        casesList, dateList = [df.loc[i][1] for i in range(14, -1, -1)], [df.loc[i][2] for i in range(14, -1, -1)]
        casesTrain = casesList[:10:]
        model = linear_model.LinearRegression()
        model.fit([[i] for i in range(0, 10)], [[i] for i in casesTrain])
        predictData = [int(i[0]) for i in model.predict([[i] for i in range(10, 15)])]
        trueData = casesList[-6:-1]
        plt.subplots(figsize=(12, 10))
        plt.plot(dateList[10:],predictData,label="预测数据")
        plt.plot(dateList[10:],trueData,label="真实数据")
        # 旋转横坐标，防止重叠
        plt.title('预测确诊人数变化')
        plt.xlabel('时间', fontsize=9)
        plt.ylabel('确诊人数', fontsize=9)
        plt.legend()  # 显示图例
        plt.show()



if __name__ == "__main__":
    matplotlib.rcParams['font.sans-serif'] = ['SimHei']
    da = DataAnalysis()
    # da.worldTrend()
    # da.newCasesTop10()
    # da.totalCasesTop10()
    # da.rateOfEachCountry()
    # da.rateTop10()
    # da.vaccineNum()
    # da.vaccineRate()
    # da.GDPTop10TotalCases()
    da.deathRateTop10()
    # da.bestTop10()
    # da.predict()
