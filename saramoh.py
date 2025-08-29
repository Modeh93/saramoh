# -*- coding: utf-8 -*-
import os, uuid
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

# ---------------- Page config
st.set_page_config(page_title="لوحة مؤشرات الصحة (PBI-Style)", page_icon="🛡️", layout="wide")

# ---------------- Palette (MOH)
PRIMARY = '#0f3d33'  # أخضر الوزارة
GOLD    = '#c6ab6e'  # ذهبي
ACCENT1 = '#8bb69b'  # أخضر فاتح
WARN    = '#ef6c00'  # تحذير
DANGER  = '#c62828'  # أحمر
MUTED   = '#9e9e9e'
SEQ = [PRIMARY, GOLD, ACCENT1, WARN, DANGER, MUTED]

# ---------------- Auth (بسيط)
if 'auth_ok' not in st.session_state:
    st.session_state.auth_ok = False

def do_login(u, p):
    return u == 'minister' and p == 'moh2025'

if not st.session_state.auth_ok:
    with st.form('login'):
        st.markdown("<h3 style='text-align:right'>تسجيل الدخول </h3>", unsafe_allow_html=True)
        u = st.text_input('اسم المستخدم')
        p = st.text_input('كلمة المرور', type='password')
        if st.form_submit_button('دخول'):
            if do_login(u, p):
                st.session_state.auth_ok = True
                st.session_state.session_id = str(uuid.uuid4())
                st.rerun()
            else:
                st.error('بيانات الدخول غير صحيحة')
    st.stop()

# ---------------- Styles (RTL + نظافة)
st.markdown("""
<style>
html, body, [class*="css"] {
  direction: rtl;
  font-family: "Noto Naskh Arabic","Tahoma","Segoe UI",sans-serif;
}
h1,h2,h3,h4 { font-weight: 700; }
div[data-testid="stMetricValue"] { color: #0f3d33; }
</style>
""", unsafe_allow_html=True)

# ---------------- Helpers
@st.cache_data(show_spinner=False)
def load_csv(src):
    return pd.read_csv(src, encoding='utf-8')

def to_datetime_naive(series):
    try:
        s = pd.to_datetime(series, errors='coerce')
        if hasattr(s, 'dt') and s.dt.tz is not None:
            s = s.dt.tz_convert(None)
        return s
    except Exception:
        return pd.to_datetime(series, errors='coerce', utc=True).dt.tz_convert(None)

def is_yes(x):
    return str(x).strip() in ['1', 'True', 'true', 'نعم', 'yes', 'Yes']

def percent(n, d):
    return (n / d * 100) if d else 0

def chunks(lst, n):
    """تقسيم قائمة إلى مجموعات حجمها n."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]

def wrap_label_ar(text, max_chars=10):
    """يلفّ النص العربي إلى أسطر قصيرة باستخدام <br> لتجنب التداخل في المحاور."""
    if not isinstance(text, str):
        text = str(text)
    words = text.split()
    lines, cur = [], ""
    for w in words:
        nxt = w if not cur else cur + " " + w
        if len(nxt) <= max_chars:
            cur = nxt
        else:
            if cur:
                lines.append(cur)
            cur = w
    if cur:
        lines.append(cur)
    return "<br>".join(lines) if lines else text

# ---------------- مراكز المحافظات (للخريطة)
GOV_CENTROIDS = {
    "دمشق": (33.514, 36.277), "ريف دمشق": (33.516, 36.5), "حلب": (36.213, 37.155),
    "إدلب": (35.930, 36.633), "اللاذقية": (35.523, 35.791), "طرطوس": (34.889, 35.886),
    "حماة": (35.131, 36.757), "حمص": (34.732, 36.723), "درعا": (32.617, 36.106),
    "القنيطرة": (33.125, 35.824), "السويداء": (32.706, 36.569), "دير الزور": (35.333, 40.150),
    "الرقة": (35.957, 39.008), "الحسكة": (36.507, 40.747),
}

def governorate_bubble_df(df):
    """فقاعات على مراكز المحافظات (لا نقرأ lat/lon من الملف)."""
    if 'governorate_entry' not in df.columns:
        return pd.DataFrame()
    grp = df.groupby('governorate_entry').size().reset_index(name='count')
    rows = []
    for _, r in grp.iterrows():
        g = r['governorate_entry']; c = int(r['count'])
        if g in GOV_CENTROIDS:
            lat, lon = GOV_CENTROIDS[g]
            rows.append({'المحافظة': g, 'lat': lat, 'lon': lon, 'count': c})
    return pd.DataFrame(rows)

# ---------------- Load data
st.sidebar.header('البيانات')
up = st.sidebar.file_uploader('ارفع CSV أو استخدم العينة', type=['csv'])

try:
    df = load_csv(up) if up else load_csv('seed_moh_dataset.csv')
except Exception as e:
    st.error(f'تعذر تحميل الملف: {e}')
    st.stop()

# Parse أنواع وتواريخ
if 'DateOfVisit' in df.columns:
    df['DateOfVisit'] = to_datetime_naive(df['DateOfVisit'])

for c in [
    'q1.9.6','q1.9.7','q3.1','q3.1_active','q3.1_not_active','q3.1_admission_avg',
    'q3.3.1_active','q3.3.1_inactive','q3.3.2_active','q3.3.2_inactive','q3.3.3_active','q3.3.3_inactive'
]:
    if c in df.columns:
        df[c] = pd.to_numeric(df[c], errors='coerce')

# ---------------- Filters (مع تحديد/إلغاء الكل)
st.sidebar.header('الفلاتر')

fid = st.sidebar.text_input('الرمز التعريفي للمنشأة')
if fid and 'facility_id' in df.columns:
    df = df[df['facility_id'].astype(str).str.contains(fid, case=False, na=False)]

fname = st.sidebar.text_input('اسم المنشأة')
if fname and 'facility_name' in df.columns:
    df = df[df['facility_name'].astype(str).str.contains(fname, case=False, na=False)]

# محافظة: multiselect + أزرار تحديد الكل/إلغاء
if 'governorate_entry' in df.columns:
    all_govs = sorted(df['governorate_entry'].dropna().unique().tolist())
    if 'sel_govs' not in st.session_state:
        st.session_state.sel_govs = all_govs.copy()
    c1, c2 = st.sidebar.columns(2)
    if c1.button('تحديد الكل'):
        st.session_state.sel_govs = all_govs.copy()
    if c2.button('إلغاء الكل'):
        st.session_state.sel_govs = []
    sel_g = st.sidebar.multiselect('المحافظة', all_govs, default=st.session_state.sel_govs, key='govs_ms')
    st.session_state.sel_govs = sel_g
    if sel_g:
        df = df[df['governorate_entry'].isin(sel_g)]

# منطقة/لواء
if 'area_entry' in df.columns:
    areas = sorted(df['area_entry'].dropna().unique())
    sel_a = st.sidebar.multiselect('المنطقة/اللواء', areas, default=[])
    if sel_a:
        df = df[df['area_entry'].isin(sel_a)]

# نطاق التاريخ
if 'DateOfVisit' in df.columns and df['DateOfVisit'].notna().any():
    mn = df['DateOfVisit'].min().date()
    mx = df['DateOfVisit'].max().date()
    dr = st.sidebar.date_input('تاريخ الزيارة', value=(mn, mx))
    if isinstance(dr, tuple) and len(dr) == 2:
        start = pd.to_datetime(dr[0])
        end = pd.to_datetime(dr[1]) + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
        df = df[(df['DateOfVisit'] >= start) & (df['DateOfVisit'] <= end)]

# حالة المنشأة
if 'facility_state' in df.columns:
    state_order = ['تعمل', 'تعمل بشكل جزئي', 'لا تعمل']
    all_states = [s for s in state_order if s in df['facility_state'].unique()]
    sel_s = st.sidebar.multiselect('حالة المنشأة', all_states, default=[])
    if sel_s:
        df = df[df['facility_state'].isin(sel_s)]

# فئة وتبعية
if 'q1.7' in df.columns:
    cats = sorted(df['q1.7'].dropna().unique())
    sel_c = st.sidebar.multiselect('فئة المنشأة', cats, default=[])
    if sel_c:
        df = df[df['q1.7'].isin(sel_c)]

if 'q1.8' in df.columns:
    deps = sorted(df['q1.8'].dropna().unique())
    sel_d = st.sidebar.multiselect('تبعية المنشأة', deps, default=[])
    if sel_d:
        df = df[df['q1.8'].isin(sel_d)]

# ---------------- KPIs
st.markdown("<h2>لوحة مؤشرات القطاع الصحي — وزارة الصحة السورية</h2>", unsafe_allow_html=True)

k1, k2, k3, k4, k5,k6 = st.columns(6)
total = len(df)
working = df['facility_state'].eq('تعمل').sum() if 'facility_state' in df.columns else 0
partial = df['facility_state'].eq('تعمل بشكل جزئي').sum() if 'facility_state' in df.columns else 0
not_work = df['facility_state'].eq('لا تعمل').sum() if 'facility_state' in df.columns else 0
licensed = df['q1.10'].apply(is_yes).sum() if 'q1.10' in df.columns else 0


k1.metric('إجمالي المنشآت 🏥', f'{total:,}')
k2.metric('العاملة ✅', f'{working:,}', f'{percent(working, total):.1f}%')
k3.metric('تعمل بشكل جزئي✅', f'{partial:,}', f'{percent(partial, total):.1f}%')
k4.metric('متوقف ⚠️', f'{not_work:,}', f'{percent(not_work, total):.1f}%')
k5.metric('مرخصة 🛡️', f'{licensed:,}', f'{percent(licensed, total):.1f}%')


st.markdown("---")

# ---------------- Tabs
tabs = st.tabs(['النظرة العامة', 'الخدمات', 'الكوادر', 'المستشفيات', 'البنية التحتية', 'السجل'])

# ===== النظرة العامة
with tabs[0]:
    # حالة المنشآت حسب المحافظة
    if 'governorate_entry' in df.columns and 'facility_state' in df.columns and len(df):
        order_states = ['تعمل', 'تعمل بشكل جزئي', 'لا تعمل']
        colors = [PRIMARY, WARN, DANGER]

        grp = (
            df[df['facility_state'].isin(order_states)]
            .groupby(['governorate_entry', 'facility_state'])
            .size()
            .reset_index(name='عدد')
        )
        pivot = grp.pivot(index='governorate_entry', columns='facility_state', values='عدد').fillna(0)
        for s in order_states:
            if s not in pivot.columns:
                pivot[s] = 0
        pivot = pivot[order_states]

        fig_state = go.Figure()
        for s, c in zip(order_states, colors):
            fig_state.add_bar(
                x=pivot.index, y=pivot[s], name=s,
                text=pivot[s], textposition='inside', insidetextanchor='middle',
                marker_color=c
            )
        fig_state.update_layout(
            title='حالة المنشآت حسب المحافظة',
            barmode='stack',
            bargap=0.15,
            bargroupgap=0.02,
            height=560,
            legend=dict(orientation='h', yanchor='bottom', y=1.18, xanchor='right', x=1),
            margin=dict(t=90, b=160, l=10, r=10),
            xaxis=dict(title='', tickangle=-25, automargin=True),
            yaxis=dict(title='عدد المنشآت'),
            uniformtext_minsize=10,
            uniformtext_mode='hide'
        )

        fig_cat = None
        if 'q1.7' in df.columns and len(df):
            cat = df['q1.7'].value_counts().reset_index()
            cat.columns = ['فئة المنشأة', 'عدد']
            fig_cat = px.pie(
                cat, names='فئة المنشأة', values='عدد', hole=0.45,
                title='توزيع فئات المنشآت', color_discrete_sequence=SEQ
            )
            fig_cat.update_layout(margin=dict(t=70, b=20))

        c1, c2 = st.columns(2)
        if fig_cat is not None:
            c1.plotly_chart(fig_cat, use_container_width=True)
        if 'fig_state' in locals():
            c2.plotly_chart(fig_state, use_container_width=True)

    st.markdown('---')

    # خريطة هيت ماب
    bubbles = governorate_bubble_df(df)
    if not bubbles.empty:
        mapfig = px.density_mapbox(
            bubbles, lat='lat', lon='lon', z='count',
            radius=55,
            center=dict(lat=bubbles['lat'].mean(), lon=bubbles['lon'].mean()),
            zoom=5.4,
            hover_name='المحافظة',
            hover_data={'count': True, 'lat': False, 'lon': False},
            color_continuous_scale=[
                [0.0, "#E6E6E6"],
                [0.25, "#00B7E1"],
                [0.5, "#005C99"],
                [0.75, "#00A99D"],
                [1.0, "#006C5B"]
            ],
            height=520
        )
        mapfig.update_layout(
            mapbox_style="open-street-map",
            margin=dict(t=10, b=10, l=10, r=10),
            coloraxis_colorbar=dict(
                title=dict(text="عدد المراكز", font=dict(size=14)),
                tickfont=dict(size=12),
                ticks="outside"
            )
        )
        st.plotly_chart(mapfig, use_container_width=True)
    else:
        st.info('لا تتوفر بيانات كافية لرسم الخريطة.')

# ===== الخدمات
with tabs[1]:
    service_map = {
        'q1.20_1': 'تنظيم أسرة', 'q1.20_2': 'رعاية حامل', 'q1.20_3': 'رعاية الأم', 'q1.20_4': 'خدمات نسائية أخرى', 'q1.20_5': 'رعاية وليد',
        'q1.20_6': 'توليد طبيعي', 'q1.20_7': 'ولادة قيصرية', 'q1.20_8': 'صحة الطفل', 'q1.20_9': 'لقاح', 'q1.20_10': 'الرعاية المتكاملة للطفل',
        'q1.20_11': 'برنامج المراهقين', 'q1.20_12': 'مسنين', 'q1.20_13': 'صحة الفم والأسنان', 'q1.20_14': 'نفسية', 'q1.20_15': 'الأمراض السارية',
        'q1.20_16': 'التشخيص المخبري', 'q1.20_17': 'صيدلية', 'q1.20_18': 'الأمراض المزمنة', 'q1.20_19': 'سكري', 'q1.20_20': 'جراحة صغرى',
        'q1.20_21': 'جراحية', 'q1.20_22': 'تنظيرية', 'q1.20_23': 'غسيل الكلى', 'q1.20_24': 'معالجة فيزيائية', 'q1.20_25': 'الأشعة',
        'q1.20_26': 'عيادة عامة', 'q1.20_27': 'اسعاف', 'q1.20_-96': 'خدمات أخرى'
    }
    present = [c for c in service_map if c in df.columns]
    if not present:
        st.warning('لم يتم العثور على أعمدة الخدمات q1.20_* في الملف.')
    else:
        # نسب لكل خدمة
        rows = []
        for c in present:
            yes = df[c].apply(is_yes).sum()
            tot = df[c].notna().sum()
            rows.append({'الخدمة': service_map[c], 'النسبة %': round(percent(yes, tot), 1), 'عدد نعم': yes, 'إجمالي': tot})
        svc = pd.DataFrame(rows).sort_values('الخدمة')

        st.subheader('جميع الخدمات')
        for group in chunks(svc, 10):
            group = group.copy()
            group['الخدمة'] = group['الخدمة'].apply(lambda x: wrap_label_ar(x, max_chars=12))

            fig = px.bar(
                group, x='الخدمة', y='النسبة %', title=None, text='النسبة %',
                color_discrete_sequence=[PRIMARY]
            )
            fig.update_traces(texttemplate='%{text:.1f}%', textposition='outside', cliponaxis=False)
            fig.update_layout(
                xaxis_tickangle=0,
                margin=dict(t=20, b=140, l=10, r=10),
                height=460, yaxis_range=[0, 100],
                xaxis=dict(automargin=True)
            )
            st.plotly_chart(fig, use_container_width=True)

        st.dataframe(svc.sort_values('النسبة %', ascending=False), use_container_width=True, hide_index=True)

        # Heatmap: الخدمات حسب المحافظة
        if 'governorate_entry' in df.columns:
            st.markdown("### جدول حراري: الخدمات حسب المحافظة")
            present_cols = [c for c in service_map if c in df.columns]
            if present_cols:
                rows = []
                for g, dfg in df.groupby('governorate_entry'):
                    for c in present_cols:
                        yes = dfg[c].apply(is_yes).sum()
                        tot = dfg[c].notna().sum()
                        pct = percent(yes, tot)
                        rows.append({'المحافظة': g, 'الخدمة': service_map[c], 'النسبة %': pct})
                heat = pd.DataFrame(rows)
                heat['الخدمة_wrapped'] = heat['الخدمة'].apply(lambda x: wrap_label_ar(x, max_chars=12))
                pivot = heat.pivot(index='المحافظة', columns='الخدمة', values='النسبة %').fillna(0)

                hm = px.imshow(
                    pivot,
                    aspect='auto',
                    color_continuous_scale=[[0.0, "#E6E6E6"], [0.25, "#8bb69b"], [0.5, "#0f3d33"], [1.0, "#c62828"]],
                    origin='upper',
                    labels=dict(color='النسبة %')
                )
                hm.update_layout(margin=dict(t=10, b=80, l=10, r=10), height=520, xaxis=dict(automargin=True))
                hm.update_traces(
                    texttemplate="%{z:.0f}%",
                    text=pivot.round(0).astype(int).values,
                    hovertemplate="المحافظة=%{y}<br>الخدمة=%{x}<br>النسبة=%{z:.1f}%<extra></extra>"
                )
                st.plotly_chart(hm, use_container_width=True)
            else:
                st.info("لا توجد أعمدة q1.20_* لحساب الجدول الحراري.")

# ===== الكوادر
with tabs[2]:
    staff_labels = {
        'q2_1': 'طبيب عام', 'q2_2': 'طبيب مقيم', 'q2_3': 'اختصاصي أسرة', 'q2_4': 'اختصاصي أطفال', 'q2_5': 'اختصاصي نسائية',
        'q2_6': 'اختصاصي داخلية', 'q2_7': 'اختصاصي قلبية', 'q2_8': 'اختصاصي جراحة عامة', 'q2_9': 'اختصاصي طوارئ', 'q2_10': 'أنف أذن حنجرة',
        'q2_11': 'عينية', 'q2_12': 'أشعة', 'q2_13': 'نفسي', 'q2_14': 'طبيب أسنان', 'q2_15': 'صيدلاني', 'q2_16': 'جلدية', 'q2_17': 'بولية',
        'q2_18': 'عظمية', 'q2_19': 'مخبر', 'q2_20': 'معالجة فيزيائية', 'q2_21': 'تمريض', 'q2_22': 'قابلات',
        'q2_23': 'فني أشعة', 'q2_24': 'فني مخبر', 'q2_25': 'فني صيدلة', 'q2_26': 'فني معالجة فيزيائية', 'q2_27': 'فني أسنان',
        'q2_28': 'فني صحة عامة', 'q2_29': 'مساعد/ة ممرض/ة', 'q2_30': 'مرشد اجتماعي', 'q2_31': 'إداري', 'q2_32': 'هندسي', 'q2_33': 'خدمي',
        'q2_34': 'فني إحصاء', 'q2_35': 'فني تخدير', 'q2_36': 'فني عمليات', 'q2_37': 'فني أطراف صناعية', 'q2_39': 'مساعد مهندس',
        'q2__96_yn_1': 'كادر آخر (1)', 'q2__96_yn_2': 'كادر آخر (2)'
    }
    present = [c for c in staff_labels if c in df.columns]
    if not present:
        st.warning('لا توجد أعمدة q2_* في الملف.')
    else:
        rows = []
        for c in present:
            yes = df[c].apply(is_yes).sum()
            tot = df[c].notna().sum()
            rows.append({'الفئة': staff_labels[c], 'النسبة %': round(percent(yes, tot), 1), 'عدد نعم': yes, 'إجمالي': tot})
        sdf = pd.DataFrame(rows).sort_values('الفئة')

        st.subheader('جميع فئات الكادر')
        for group in chunks(sdf, 10):
            group = group.copy()
            group['الفئة_wrapped'] = group['الفئة'].apply(lambda x: wrap_label_ar(x, max_chars=12))

            fig = px.bar(
                group, x='الفئة_wrapped', y='النسبة %', title=None, text='النسبة %', color_discrete_sequence=[GOLD]
            )
            fig.update_traces(texttemplate='%{text:.1f}%', textposition='outside', cliponaxis=False)
            fig.update_layout(
                xaxis_tickangle=0,
                margin=dict(t=20, b=140, l=10, r=10),
                height=460, yaxis_range=[0, 100],
                xaxis=dict(automargin=True)
            )
            st.plotly_chart(fig, use_container_width=True)

        st.dataframe(sdf.sort_values('النسبة %', ascending=False), use_container_width=True, hide_index=True)

        # Heatmap: الكوادر حسب المحافظة
        if 'governorate_entry' in df.columns:
            st.markdown("### جدول حراري: الكوادر حسب المحافظة")
            present_staff = [c for c in staff_labels if c in df.columns]
            if present_staff:
                rows = []
                for g, dfg in df.groupby('governorate_entry'):
                    for c in present_staff:
                        yes = dfg[c].apply(is_yes).sum()
                        tot = dfg[c].notna().sum()
                        pct = percent(yes, tot)
                        rows.append({'المحافظة': g, 'الفئة': staff_labels[c], 'النسبة %': pct})
                sheat = pd.DataFrame(rows)
                sheat['الفئة_wrapped'] = sheat['الفئة'].apply(lambda x: wrap_label_ar(x, max_chars=10))
                spivot = sheat.pivot(index='المحافظة', columns='الفئة_wrapped', values='النسبة %').fillna(0)

                shm = px.imshow(
                    spivot,
                    aspect='auto',
                    color_continuous_scale=[[0.0, "#E6E6E6"], [0.25, GOLD], [0.6, PRIMARY], [1.0, DANGER]],
                    origin='upper',
                    labels=dict(color='النسبة %')
                )
                shm.update_layout(margin=dict(t=10, b=80, l=10, r=10), height=520, xaxis=dict(automargin=True))
                shm.update_traces(
                    texttemplate="%{z:.0f}%",
                    text=spivot.round(0).astype(int).values,
                    hovertemplate="المحافظة=%{y}<br>الفئة=%{x}<br>النسبة=%{z:.1f}%<extra></extra>"
                )
                st.plotly_chart(shm, use_container_width=True)
            else:
                st.info("لا توجد أعمدة q2_* لحساب الجدول الحراري.")

# ===== المستشفيات
with tabs[3]:
    c1, c2, c3, c4 = st.columns(4)
    total_beds   = df['q3.1'].sum() if 'q3.1' in df.columns else 0
    active_beds  = df['q3.1_active'].sum() if 'q3.1_active' in df.columns else 0
    inact_beds   = df['q3.1_not_active'].sum() if 'q3.1_not_active' in df.columns else 0
    los          = df['q3.1_admission_avg'].mean() if 'q3.1_admission_avg' in df.columns else 0

    c1.metric('مجموع الأسرة', f'{int(total_beds):,}')
    c2.metric('الأسرة الفعّالة', f'{int(active_beds):,}')
    c3.metric('الأسرة غير الفعّالة', f'{int(inact_beds):,}')
    c4.metric('متوسط أيام الإقامة', f'{los:.1f}')

    for a, i, title in [
        ('q3.3.1_active', 'q3.3.1_inactive', 'أسرة العناية المشددة'),
        ('q3.3.2_active', 'q3.3.2_inactive', 'المنافس'),
        ('q3.3.3_active', 'q3.3.3_inactive', 'الحواضن')
    ]:
        if a in df.columns and i in df.columns:
            sums = pd.DataFrame({'الحالة': ['فعّالة', 'معطلة'], 'العدد': [df[a].sum(), df[i].sum()]})
            bf = px.bar(
                sums, y='الحالة', x='العدد', color='الحالة', orientation='h',
                title=title, color_discrete_sequence=[PRIMARY, GOLD]
            )
            bf.update_traces(text=sums['العدد'], textposition='outside', cliponaxis=False)
            bf.update_layout(
                height=300, bargap=0.45, margin=dict(t=60, b=40, l=10, r=10),
                legend=dict(orientation='h', yanchor='bottom', y=1.05, xanchor='left', x=0)
            )
            st.plotly_chart(bf, use_container_width=True)

# ===== البنية التحتية
with tabs[4]:
    m1, m2 = st.columns(2)
    if 'q4.8' in df.columns:
        yes = df['q4.8'].apply(is_yes).sum(); tot = df['q4.8'].notna().sum()
        m1.metric('⚡ الكهرباء متوفرة', f'{percent(yes, tot):.1f}%')
    if 'q4.5' in df.columns:
        yes = df['q4.5'].apply(is_yes).sum(); tot = df['q4.5'].notna().sum()
        m2.metric('🚑 توفر سيارة إسعاف', f'{percent(yes, tot):.1f}%')

    c1, c2 = st.columns(2)
    if 'q4.18' in df.columns:
        water = df['q4.18'].value_counts().reset_index()
        water.columns = ['مصدر المياه', 'عدد']
        ip = px.pie(water, names='مصدر المياه', values='عدد', hole=0.45, title='مصادر المياه الرئيسية', color_discrete_sequence=SEQ)
        c1.plotly_chart(ip, use_container_width=True)

    if 'q4.17' in df.columns:
        duty = df['q4.17'].value_counts().reset_index()
        duty.columns = ['فترة الدوام', 'عدد']
        ib = px.bar(duty, x='فترة الدوام', y='عدد', title='فترة الدوام في المنشآت', color_discrete_sequence=[PRIMARY])
        ib.update_layout(margin=dict(t=60, b=80))
        c2.plotly_chart(ib, use_container_width=True)

# ===== السجل
with tabs[5]:
    st.subheader('سجل المنشآت (تفصيلي)')
    col_map = {
        'facility_id': 'الرمز التعريفي للمنشأة',
        'facility_name': 'اسم المنشأة',
        'DateOfVisit': 'تاريخ الزيارة',
        'enumerator_id': 'الرقم التعريفي للباحث',
        'enumerator_name': 'اسم الباحث',
        'governorate_entry': 'المحافظة',
        'area_entry': 'المنطقة/اللواء',
        'facility_state': 'حالة المنشأة',
        'q1.7': 'فئة المنشأة',
        'q1.8': 'تبعية المنشأة',
        'q1.10': 'ترخيص',
        'q1.9.6': 'عدد المراجعين (آخر شهر)',
        'q1.9.7': 'عدد السكان المخدّمين تقريباً'
    }
    show_cols = [c for c in col_map if c in df.columns]
    df_show = df[show_cols].rename(columns=col_map) if show_cols else pd.DataFrame()

    sort_col = 'تاريخ الزيارة' if 'تاريخ الزيارة' in df_show.columns else (df_show.columns[0] if len(df_show.columns) else None)
    if sort_col:
        df_show = df_show.sort_values(by=sort_col, ascending=False)

    st.dataframe(df_show, use_container_width=True, hide_index=True)

    st.download_button(
        'تنزيل الملف بعد التصفية (CSV)',
        data=df.to_csv(index=False).encode('utf-8-sig'),
        file_name='filtered_moh_dataset.csv',
        mime='text/csv'
    )
