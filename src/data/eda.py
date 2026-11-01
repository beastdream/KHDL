"""Train-only EDA; seven figures and Vietnamese evidence-based interpretation."""
import json
import inspect
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from src.data.schema import NUMERIC_FEATURES, TARGET
from src.data.week2_inputs import load_split, install_data_guard
from src.utils.paths import REPORTS, write_json


def run_eda():
    df=load_split('train')
    folder=REPORTS/'figures'
    paths=[]
    sns.set_theme(style='whitegrid')
    # Seaborn 0.13 forwards a deprecated bool; opt into the modern bxp API.
    # Older Matplotlib without orientation retains its original supported call.
    box_options = ({'vert': None, 'orientation': 'vertical'}
                   if 'orientation' in inspect.signature(matplotlib.axes.Axes.bxp).parameters else {})
    def save(name):
        path=folder/f'week2_{name}.png'
        plt.tight_layout()
        plt.savefig(path,dpi=160,bbox_inches='tight')
        plt.close()
        paths.append(str(path.relative_to(REPORTS)))
    plt.figure(figsize=(9,5))
    sns.histplot(df.SalePrice,bins=35,kde=True)
    plt.title('Train: SalePrice distribution'); plt.xlabel('SalePrice (USD)')
    save('01_price_histogram')
    plt.figure(figsize=(9,5))
    sns.scatterplot(data=df,x='GrLivArea',y=TARGET,alpha=.5)
    plt.title('Train: living area and price'); plt.xlabel('Living area (sq ft)')
    save('02_area_price')
    plt.figure(figsize=(9,5))
    sns.boxplot(data=df,x='OverallQual',y=TARGET,**box_options)
    plt.title('Train: overall quality and price')
    save('03_quality_price')
    grouped=df.groupby('Neighborhood').SalePrice.agg(['median','count']).sort_values('median')
    plt.figure(figsize=(11,8))
    plt.barh(grouped.index,grouped['median'])
    plt.xlabel('Median SalePrice (USD)'); plt.title('Train: neighborhood medians (unadjusted)')
    save('04_neighborhood_median')
    corr=df[NUMERIC_FEATURES+[TARGET]].corr()
    plt.figure(figsize=(10,8))
    # Symmetric limits already center zero; avoid redundant Seaborn recentering.
    sns.heatmap(corr,annot=True,fmt='.2f',cmap='vlag',vmin=-1,vmax=1)
    plt.title('Train: Pearson correlations')
    save('05_correlation')
    plt.figure(figsize=(9,5))
    sns.scatterplot(data=df,x='YearBuilt',y=TARGET,alpha=.5)
    plt.title('Train: construction year and price')
    save('06_year_price')
    plt.figure(figsize=(9,5))
    sns.boxplot(data=df,x='BedroomAbvGr',y=TARGET,**box_options)
    plt.title('Train: above-ground bedrooms and price')
    save('07_bedrooms_price')
    q1,q3=df.SalePrice.quantile([.25,.75])
    out=df.loc[(df.SalePrice<q1-1.5*(q3-q1))|(df.SalePrice>q3+1.5*(q3-q1)),['Id','GrLivArea',TARGET]]
    areas=df.loc[df.GrLivArea>4000,['Id','GrLivArea',TARGET]]
    correlations=corr[TARGET].drop(TARGET).sort_values(ascending=False)
    result={'split':'train','rows':len(df),'target_median':float(df.SalePrice.median()),'target_skew':float(df.SalePrice.skew()),
            'pearson_target':correlations.to_dict(),'neighborhoods':grouped.reset_index().to_dict('records'),
            'iqr_outliers':out.to_dict('records'),'large_area_outliers':areas.to_dict('records'), 'figures':paths,'rows_removed':0}
    write_json(REPORTS/'eda_summary.json',result)
    lines=['# Phân tích EDA — chỉ Train', '', f"Phân tích {len(df)} mẫu Train; không đọc Validation hoặc Test để EDA.", '',
           '## Xu hướng chính', '', f"SalePrice có skewness {result['target_skew']:.3f}, trung vị {result['target_median']:,.0f} USD. Phân phối lệch phải: một số nhà giá cao kéo dài đuôi phân phối.",
           'Diện tích sinh hoạt, chất lượng tổng thể và năm xây dựng có liên hệ với giá; biểu đồ cho thấy độ phân tán đáng kể giữa các căn nhà.', '',
           '## Tương quan với giá', '', '| Biến | Pearson r với SalePrice |', '|---|---:|']
    lines += [f'| {c} | {v:.4f} |' for c,v in correlations.items()]
    lines += ['', f"Biến có tương quan tuyến tính dương lớn nhất trong 7 biến số là {correlations.index[0]} ({correlations.iloc[0]:.3f}).",
              'Số phòng ngủ đơn lẻ không đại diện đầy đủ cho diện tích/chất lượng; không suy ra tăng phòng ngủ sẽ làm tăng giá tương ứng.', '',
              '## Neighborhood', '', '| Neighborhood | Trung vị USD | Số mẫu Train |', '|---|---:|---:|']
    lines += [f'| {name} | {r["median"]:,.0f} | {int(r["count"])} |' for name,r in grouped.iterrows()]
    lines += ['', 'Giá trung vị theo khu vực là so sánh thô, chưa kiểm soát diện tích, chất lượng và năm xây dựng. Nhóm ít mẫu có ước lượng kém ổn định.', '',
              '## Outlier', '', f'Có {len(out)} mẫu ngoài ngưỡng 1.5 IQR của giá và {len(areas)} mẫu GrLivArea > 4.000 sq ft. Đây là cờ xem xét, không phải bằng chứng dữ liệu sai.',
              'Các mẫu diện tích lớn: ' + json.dumps(areas.to_dict('records'),ensure_ascii=False),
              'Danh sách ID outlier giá: ' + str(out.Id.tolist()),
              'Không xóa hoặc winsorize bất kỳ mẫu nào.', '', '## Hạn chế diễn giải', '',
              'Tương quan không chứng minh quan hệ nhân quả. Pearson chỉ đo liên hệ tuyến tính, nhạy với outlier; các features có thể tương quan lẫn nhau.',
              'OverallQual là thang thứ bậc nên hệ số Pearson chỉ mang tính mô tả. Neighborhood là biến phân loại, không mã hóa thành số thứ tự để tính Pearson.',
              'Ames là một thị trường trong một giai đoạn lịch sử; kết quả không tự động khái quát sang địa phương, thời điểm hoặc thị trường Việt Nam.',
              'EDA chỉ dựa trên Train. Missing values chưa được điền trong biểu đồ; preprocessing về sau chỉ học thống kê Train. Không dùng EDA để loại mẫu test hoặc chỉnh split.', '', '## Biểu đồ']
    lines += [f'\n![Train plot]({p})' for p in paths]
    (REPORTS/'eda_analysis.md').write_text('\n'.join(lines),encoding='utf-8')
    return result


if __name__=='__main__':
    install_data_guard()
    print(json.dumps(run_eda(),indent=2,ensure_ascii=False))
