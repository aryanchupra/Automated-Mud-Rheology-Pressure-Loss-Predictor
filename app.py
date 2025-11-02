# streamlit_step5_dashboard.py
import streamlit as st
import pandas as pd
import numpy as np
import math
from pathlib import Path
from scipy.optimize import least_squares, curve_fit
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(layout="wide", page_title="Rheology → Pressure Loss Dashboard", initial_sidebar_state="expanded")
# after: st.set_page_config(...)
st.markdown(
    """
    <div style="text-align: center; padding: 6px 0 6px 0;">
      <h1 style="margin:0; font-size:28px;">Automated Mud Rheology 💧 &amp; Pressure-Loss Predictor 📊</h1>
      <p style="margin:4px 0 0 0; color: #555;">Upload rheometer, velocity & exp. pressure data → compare rheology models and predicted ΔP</p>
    </div>
    <hr style="margin-top:10px; margin-bottom:18px;">
    """,
    unsafe_allow_html=True
)


# ----------------- Utility functions (adapted from previous steps) -----------------
def detect_columns(df):
    def find_col(keywords):
        for c in df.columns:
            lc = str(c).lower()
            for k in keywords:
                if k in lc:
                    return c
        return None
    n_col = find_col(['n (rpm)','n(rpm)','rpm',' n ','n ' ,'n'])
    theta_col = find_col(['θ','theta','dial'])
    tau_col = find_col(['tau','shear stress','τ','τ (shear stress)','tau (pa)'])
    vel_col = find_col(['vel','velocity','v (m/s)','velocity (m/s)'])
    dp_col = find_col(['dp','dP','pressure drop','pressure','dP_exp'])
    return n_col, theta_col, tau_col, vel_col, dp_col

def preprocess_df(df):
    n_col, theta_col, tau_col, _, _ = detect_columns(df)
    if tau_col is None and theta_col is not None:
        df['tau'] = 0.51 * pd.to_numeric(df[theta_col], errors='coerce')
    elif tau_col is not None:
        df['tau'] = pd.to_numeric(df[tau_col], errors='coerce')
    else:
        raise ValueError("No 'tau' or 'theta' column found.")
    if n_col is None:
        for c in df.columns:
            try:
                s = pd.to_numeric(df[c], errors='coerce').dropna()
                if len(s)>3 and s.between(1,2000).sum()>2:
                    n_col = c; break
            except Exception:
                pass
        if n_col is None:
            raise ValueError("No N/RPM column found.")
    df['N'] = pd.to_numeric(df[n_col], errors='coerce')
    df = df.dropna(subset=['N','tau']).reset_index(drop=True)
    return df, n_col, theta_col, tau_col

def fit_kmt0_from_arrays(N, tau, initial_guess=(0.05,0.5,0.0)):
    N = np.asarray(N, dtype=float); tau = np.asarray(tau, dtype=float)
    omega = 2 * math.pi * N / 60.0
    min_tau = np.min(tau)
    lower = np.array([1e-12, 1e-12, -np.inf]); upper = np.array([np.inf, np.inf, min_tau - 1e-9])
    def resid(params):
        k, m, t0 = params; diff = tau - t0
        if np.any(diff <= 0) or k <= 0 or m <= 0:
            return 1e6 * np.ones_like(omega)
        pred = k * np.power(diff, m)
        return np.log(omega) - np.log(pred)
    res = least_squares(resid, x0=np.array(initial_guess), bounds=(lower, upper),
                        ftol=1e-12, xtol=1e-12, gtol=1e-12, max_nfev=20000)
    k_fit, m_fit, t0_fit = res.x
    omega_pred = k_fit * np.power(tau - t0_fit, m_fit)
    return {'k':float(k_fit),'m':float(m_fit),'tau0':float(t0_fit),'success':bool(res.success)}, k_fit, m_fit, t0_fit

def gamma_improved_formula(tau_val, mu_p, k, m, tau0, s, beta):
    tau = float(tau_val); mu = float(mu_p); diff = tau - tau0
    if diff <= 0 or s <= 0 or mu <= 0: return float('nan')
    termA = k * (diff**m)
    termB = (tau0 / mu) * math.log(s)
    num = 1.5*(beta**3) * (termA - termB) - ((1.5*(beta**2)+1.5*beta+2.0)*(tau0/mu))
    denom = (m*(m-1)*tau**2)/(diff**2) - (1.5*beta+3.0)*m*tau/diff + (1.5*(beta**2)+1.5*beta+2.0)
    if denom == 0 or not np.isfinite(denom): return float('nan')
    return float(num/denom)

def solve_mu_p_secant(theta600, theta300, k, m, tau0, s, beta, df,
                      x0=1e-6, x1=0.1, tol=1e-10, max_iter=500):
    def tau_at_N(Nval): return df.loc[(df['N']-Nval).abs().idxmin(),'tau']
    def RHS(mu):
        tau6 = tau_at_N(600); tau3 = tau_at_N(300)
        g6 = gamma_improved_formula(tau6, mu, k, m, tau0, s, beta)
        g3 = gamma_improved_formula(tau3, mu, k, m, tau0, s, beta)
        if not np.isfinite(g6) or not np.isfinite(g3) or (g6-g3)==0: return float('nan')
        return 0.51*(theta600-theta300)/(g6-g3)
    def F(mu):
        rhs = RHS(mu); return mu-rhs if np.isfinite(rhs) else float('nan')
    f0, f1 = F(x0), F(x1)
    if not np.isfinite(f0) or not np.isfinite(f1):
        x0, x1 = 1e-8, 1.0; f0, f1 = F(x0), F(x1)
    for i in range(max_iter):
        if (f1-f0)==0 or not np.isfinite(f0) or not np.isfinite(f1):
            x0, x1 = 1e-8, 1.0; f0, f1 = F(x0), F(x1)
        x2 = x1 - f1*(x1-x0)/(f1-f0); f2 = F(x2)
        if np.isfinite(f2) and (abs(f2)<tol or abs(x2-x1)<tol): return x2
        x0,f0,x1,f1 = x1,f1,x2,f2
    return x1

# Rheology models and fits
def bingham(g, tau_y, mu_p): return tau_y + mu_p*g
def powerlaw(g, K, n): return K*(g**n)
def hb(g, tau0, K, n): return tau0 + K*(g**n)
def casson(g, tau0_c, mu_c): return (np.sqrt(np.abs(tau0_c)) + np.sqrt(np.abs(mu_c*g)))**2
def rs(g, A, C, n): return A*((g+np.abs(C))**n)

def fit_bingham(gamma, tau):
    A = np.vstack([np.ones_like(gamma), gamma]).T
    coef = np.linalg.lstsq(A, tau, rcond=None)[0]
    tau_y, mu_p = coef[0], coef[1]
    pred = bingham(gamma, tau_y, mu_p); mse = np.mean((tau-pred)**2)
    return {'tau_y':float(tau_y),'mu_p':float(mu_p),'mse':float(mse),'pred':pred}

def fit_powerlaw(gamma, tau):
    mask = (gamma>0)&(tau>0)
    if mask.sum()<3: return None
    slope, intercept = np.polyfit(np.log(gamma[mask]), np.log(tau[mask]),1)
    n = float(slope); K = float(np.exp(intercept))
    pred = powerlaw(gamma, K, n); mse = np.mean((tau-pred)**2)
    return {'K':K,'n':n,'mse':float(mse),'pred':pred}

def fit_hb(gamma, tau):
    p0=[max(0.0,np.min(tau)*0.1),1.0,0.5]; bounds=([0.0,0.0,0.0],[np.max(tau)*2.0,np.inf,2.0])
    try:
        popt,_=curve_fit(hb,gamma,tau,p0=p0,bounds=bounds,maxfev=20000)
        pred=hb(gamma,*popt); mse=np.mean((tau-pred)**2)
        return {'tau0':float(popt[0]),'K':float(popt[1]),'n':float(popt[2]),'mse':float(mse),'pred':pred}
    except Exception:
        return None

def fit_casson(gamma, tau):
    p0=[max(0.0,np.min(tau)*0.1),0.01]; bounds=([0.0,0.0],[np.max(tau)*2.0,np.inf])
    try:
        popt,_=curve_fit(casson,gamma,tau,p0=p0,bounds=bounds,maxfev=20000)
        pred=casson(gamma,*popt); mse=np.mean((tau-pred)**2)
        return {'tau0_c':float(popt[0]),'mu_c':float(popt[1]),'mse':float(mse),'pred':pred}
    except Exception:
        return None

def fit_rs(gamma, tau):
    p0=[1.0,0.1,0.5]; bounds=([0.0,-np.inf,0.0],[np.inf,np.inf,5.0])
    try:
        popt,_=curve_fit(rs,gamma,tau,p0=p0,bounds=bounds,maxfev=20000)
        pred=rs(gamma,*popt); mse=np.mean((tau-pred)**2)
        return {'A':float(popt[0]),'C':float(popt[1]),'n':float(popt[2]),'mse':float(mse),'pred':pred}
    except Exception:
        return None

# Rabinowitsch + ΔP calculators
def rabinowitsch_gamma(V, D, n): return (8*V/D) * (3*n + 1)/(4*n)
def deltaP_from_tauw(tauw, L, D): return 4.0 * tauw * L / D
def deltaP_bingham_pa(V, D, L, tau_y, mu_p): return 32.0 * mu_p * L * V / (D**2) + 4.0 * tau_y * L / D
def deltaP_powerlaw_pa(V, D, L, K, n): gw = rabinowitsch_gamma(V,D,n); tauw = K*(gw**n); return deltaP_from_tauw(tauw,L,D), tauw
def deltaP_hb_Cc_pa(V, D, L, tau0, K, n):
    gw = rabinowitsch_gamma(V,D,n); denom = tau0 + K*(gw**n)
    Cc = 1.0 if denom==0 else max(0.0, min(1.0, 1.0 - (1.0/(2.0*n+1.0))*(tau0/denom)))
    tauw = tau0 + K * ((Cc * gw)**n)
    return deltaP_from_tauw(tauw,L,D), tauw, gw, Cc
def deltaP_robertson_pa(V, D, L, A, C, B): gw = rabinowitsch_gamma(V,D,B); tauw = A*((gw+abs(C))**B); return deltaP_from_tauw(tauw,L,D), tauw, gw
def deltaP_casson_pa(V, D, L, tau0c, mu_c): tauw = (math.sqrt(max(tau0c,0.0)) + math.sqrt(max(mu_c*(8.0*V/D),0.0)))**2; return deltaP_from_tauw(tauw,L,D), tauw

# ---------------------- Sidebar controls ----------------------
st.sidebar.title("Inputs & Options")
uploaded = st.sidebar.file_uploader("Upload single Excel (.xlsx) containing rheometer + velocities", type=['xlsx'])
use_sample = st.sidebar.checkbox("Use sample_project_2.xlsx (workspace) if no upload", value=True)
excel_source = uploaded if uploaded is not None else '/mnt/data/sample_project_2.xlsx' if use_sample else None
L = st.sidebar.number_input("Pipe length L (m)", value=10.97, format="%.5f")
D = st.sidebar.number_input("Pipe diameter D (m)", value=0.0508, format="%.5f")
rho = st.sidebar.number_input("Fluid density ρ (kg/m³)", value=1000.0, format="%.2f")
run_button = st.sidebar.button("Run (process file)")

if excel_source is None:
    st.info("Upload a file or enable the workspace sample option to proceed.")
    st.stop()

# ---------------------- Run pipeline when requested ----------------------
if run_button:
    try:
        # Load excel
        if isinstance(excel_source, str):
            xls = pd.ExcelFile(excel_source)
            sheet0 = xls.sheet_names[0]
            df_raw = xls.parse(sheet0)
        else:
            df_raw = pd.read_excel(excel_source)
    except Exception as e:
        st.error(f"Failed to read Excel: {e}")
        st.stop()

    # top row: detected columns summary
    n_col, theta_col, tau_col, vel_col, dp_col = detect_columns(df_raw)
    col_info_left, col_info_mid, col_info_right = st.columns(3)
    col_info_left.metric("Rheometer N column", f"{n_col}")
    col_info_mid.metric("Rheometer θ column (if present)", f"{theta_col}")
    col_info_right.metric("Rheometer τ column (if present)", f"{tau_col}")
    st.write("Detected velocity column candidate:", vel_col, "Detected dP column candidate:", dp_col)
    st.write("---")

    # Preprocess
    try:
        df_proc, ncol_detected, theta_col_detected, tau_col_detected = preprocess_df(df_raw)
    except Exception as e:
        st.error(f"Preprocess error: {e}")
        st.stop()

    # Step 1: fit k,m,tau0
    st.header("Step 1 — Fit k, m, τ₀")
    diag, k_fit, m_fit, tau0_fit = fit_kmt0_from_arrays(df_proc['N'].values, df_proc['tau'].values)
    # show as metrics
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("k", f"{diag['k']:.6g}")
    m2.metric("m", f"{diag['m']:.6g}")
    m3.metric("τ₀", f"{diag['tau0']:.6g}")
    m4.metric("Fit success", "Yes" if diag.get('success') else "No")

    st.write("Quick data preview (rheometer):")
    st.dataframe(df_proc[['N','tau']].head(10))

    # Step 2: compute mu_p via secant solve using theta600-theta300 (if theta exists)
    st.header("Step 2 — Solve μₚ and compute improved γ̇")
    S_VAL = 0.93646; BETA_VAL = 2.0/(1.0 - S_VAL**2)
    theta600 = None; theta300 = None
    if theta_col_detected:
        theta600 = df_raw[theta_col_detected].loc[(df_raw[ncol_detected]-600).abs().idxmin()]
        theta300 = df_raw[theta_col_detected].loc[(df_raw[ncol_detected]-300).abs().idxmin()]
    st.write("theta600:", theta600, "theta300:", theta300)
    mu_p = solve_mu_p_secant(theta600, theta300, k_fit, m_fit, tau0_fit, S_VAL, BETA_VAL, df_proc)
    st.metric("Solved μₚ (Pa·s)", f"{mu_p:.6g}", delta=f"{mu_p*1000.0:.3f} cP (in brackets cP)")

    # ----- NEW: compute & display conventional and improved shear rates immediately after mu_p -----
    df_proc['gamma_conv'] = 1.7 * df_proc['N']
    df_proc['gamma_imp'] = df_proc['tau'].apply(lambda t: gamma_improved_formula(t, mu_p, k_fit, m_fit, tau0_fit, S_VAL, BETA_VAL))
    st.subheader("Conventional & Improved shear rates (sample rows)")
    # show a useful subset of columns: N, tau, conventional gamma, improved gamma
    display_cols = ['N','tau','gamma_conv','gamma_imp']
    # guard against missing columns
    display_cols = [c for c in display_cols if c in df_proc.columns]
    st.dataframe(df_proc[display_cols].head(20))

    # Step 3: compute gamma conv and improved, fit rheology models
    # (these assignments are idempotent; already computed above)
    df_proc['gamma_conv'] = 1.7 * df_proc['N']
    df_proc['gamma_imp'] = df_proc['tau'].apply(lambda t: gamma_improved_formula(t, mu_p, k_fit, m_fit, tau0_fit, S_VAL, BETA_VAL))

    st.header("Step 3 — Fit rheological models (τ vs γ̇)")
    fits_conv = {'Bingham': fit_bingham(df_proc['gamma_conv'].values, df_proc['tau'].values),
                 'PowerLaw': fit_powerlaw(df_proc['gamma_conv'].values, df_proc['tau'].values),
                 'HerschelBulkley': fit_hb(df_proc['gamma_conv'].values, df_proc['tau'].values),
                 'Casson': fit_casson(df_proc['gamma_conv'].values, df_proc['tau'].values),
                 'RobertsonStiff': fit_rs(df_proc['gamma_conv'].values, df_proc['tau'].values)}
    st.subheader("Conventional γ̇ fits (summary)")
    st.json({k:v for k,v in fits_conv.items() if v is not None})

    mask_imp = (df_proc['gamma_imp']>0) & np.isfinite(df_proc['gamma_imp'])
    fits_imp = None
    if mask_imp.sum()>3:
        fits_imp = {'Bingham': fit_bingham(df_proc.loc[mask_imp,'gamma_imp'].values, df_proc.loc[mask_imp,'tau'].values),
                    'PowerLaw': fit_powerlaw(df_proc.loc[mask_imp,'gamma_imp'].values, df_proc.loc[mask_imp,'tau'].values),
                    'HerschelBulkley': fit_hb(df_proc.loc[mask_imp,'gamma_imp'].values, df_proc.loc[mask_imp,'tau'].values),
                    'Casson': fit_casson(df_proc.loc[mask_imp,'gamma_imp'].values, df_proc.loc[mask_imp,'tau'].values),
                    'RobertsonStiff': fit_rs(df_proc.loc[mask_imp,'gamma_imp'].values, df_proc.loc[mask_imp,'tau'].values)}
        st.subheader("Improved γ̇ fits (summary)")
        st.json({k:v for k,v in fits_imp.items() if v is not None})
    else:
        st.info("Not enough valid improved γ̇ points to fit improved-γ models.")

    # Step 4: predictions using velocities
    st.header("Step 4 — Predict ΔP (per model) from experimental velocities")
    # detect velocity column
    vel_candidates = [c for c in df_raw.columns if 'vel' in str(c).lower() or 'velocity' in str(c).lower()]
    if not vel_candidates:
        st.error("No velocity column found in the uploaded file. Include a column named like 'Velocity (m/s)'.")
        st.stop()
    vel_col = vel_candidates[0]
    df_exp = df_raw[[vel_col]].rename(columns={vel_col:'V'}).copy()
    dp_candidates = [c for c in df_raw.columns if 'dp' in str(c).lower() or 'pressure' in str(c).lower() or 'dP' in str(c)]
    if dp_candidates:
        dp_col = dp_candidates[0]; df_exp['dP_exp_kPa'] = pd.to_numeric(df_raw[dp_col], errors='coerce')
    else:
        df_exp['dP_exp_kPa'] = np.nan

    # choose parameter sets to evaluate
    evaluate_imp = st.checkbox("Compute predictions for IMPROVED-fit parameters (recommended)", value=True)
    evaluate_conv = st.checkbox("Also compute predictions for CONVENTIONAL-fit parameters", value=False)

    results = []
    if evaluate_conv:
        for i,row in df_exp.iterrows():
            V = float(row['V']); rec = {'param_set':'conventional','V':V,'index':i,'dP_exp_kPa':row.get('dP_exp_kPa',None)}
            conv = fits_conv
            # Bingham
            b = conv.get('Bingham')
            rec['dP_Bingham_kPa'] = deltaP_bingham_pa(V,D,L,b['tau_y'],b['mu_p'])/1000.0 if b else None
            # Power
            p = conv.get('PowerLaw')
            if p: dP_pa,tauw = deltaP_powerlaw_pa(V,D,L,p['K'],p['n']); rec['dP_PowerLaw_kPa']=dP_pa/1000.0
            # HB
            hbpar = conv.get('HerschelBulkley')
            if hbpar: dP_pa,tauw,gw,Cc = deltaP_hb_Cc_pa(V,D,L,hbpar['tau0'],hbpar['K'],hbpar['n']); rec['dP_HB_Cc_kPa']=dP_pa/1000.0
            # Casson
            cas = conv.get('Casson')
            if cas: dP_pa,tauw = deltaP_casson_pa(V,D,L,cas['tau0_c'],cas['mu_c']); rec['dP_Casson_kPa']=dP_pa/1000.0
            # RS
            rspar = conv.get('RobertsonStiff')
            if rspar: dP_pa,tauw,gw = deltaP_robertson_pa(V,D,L,rspar['A'],rspar['C'],rspar['n']); rec['dP_RS_kPa']=dP_pa/1000.0
            results.append(rec)

    if evaluate_imp and fits_imp:
        for i,row in df_exp.iterrows():
            V = float(row['V']); rec = {'param_set':'improved','V':V,'index':i,'dP_exp_kPa':row.get('dP_exp_kPa',None)}
            imp = fits_imp
            b = imp.get('Bingham'); rec['dP_Bingham_kPa'] = deltaP_bingham_pa(V,D,L,b['tau_y'],b['mu_p'])/1000.0 if b else None
            p = imp.get('PowerLaw')
            if p: dP_pa,tauw = deltaP_powerlaw_pa(V,D,L,p['K'],p['n']); rec['dP_PowerLaw_kPa']=dP_pa/1000.0
            hbpar = imp.get('HerschelBulkley')
            if hbpar: dP_pa,tauw,gw,Cc = deltaP_hb_Cc_pa(V,D,L,hbpar['tau0'],hbpar['K'],hbpar['n']); rec['dP_HB_Cc_kPa']=dP_pa/1000.0
            cas = imp.get('Casson')
            if cas: dP_pa,tauw = deltaP_casson_pa(V,D,L,cas['tau0_c'],cas['mu_c']); rec['dP_Casson_kPa']=dP_pa/1000.0
            rspar = imp.get('RobertsonStiff')
            if rspar: dP_pa,tauw,gw = deltaP_robertson_pa(V,D,L,rspar['A'],rspar['C'],rspar['n']); rec['dP_RS_kPa']=dP_pa/1000.0
            results.append(rec)

    outdf = pd.DataFrame(results)
    st.subheader("Predictions table (first rows)")
    st.dataframe(outdf.head(20))

    # ------------------ Consensus-based recommendation (improved only) ------------------
    st.header("Consensus-based Recommendation (IMPROVED parameter set)")
    if 'improved' not in outdf['param_set'].unique():
        st.info("Improved parameter set not available or not selected.")
    else:
        sub_imp = outdf[outdf['param_set']=='improved'].reset_index(drop=True)
        # model prediction columns (exclude dP_exp_kPa)
        model_cols = [c for c in sub_imp.columns if c.startswith('dP_') and c.endswith('_kPa') and c!='dP_exp_kPa']
        if not model_cols:
            st.info("No model prediction columns for improved set.")
        else:
            # ensemble mean across model predictions (row-wise)
            ensemble_mean = sub_imp[model_cols].mean(axis=1, skipna=True)
            # compute MSE of each model vs ensemble
            mse_vs_ensemble = {}
            for col in model_cols:
                mask = sub_imp[col].notna() & np.isfinite(ensemble_mean)
                mse_vs_ensemble[col] = float(((sub_imp.loc[mask,col] - ensemble_mean.loc[mask])**2).mean()) if mask.sum()>0 else None
            # sort
            valid_mses = {k:v for k,v in mse_vs_ensemble.items() if v is not None}
            if not valid_mses:
                st.info("No valid model predictions to compare.")
            else:
                best_col, best_mse = min(valid_mses.items(), key=lambda kv: kv[1])
                # Pretty summary cards
                col_a, col_b, col_c = st.columns([1,1,2])
                col_a.metric("Best model (improved set)", best_col.replace("dP_","").replace("_kPa",""))
                col_b.metric("MSE vs ensemble (kPa²)", f"{best_mse:.6g}")
                mean_pred_val = float(sub_imp[best_col].mean(skipna=True))
                median_pred_val = float(sub_imp[best_col].median(skipna=True))
                col_c.metric("Mean predicted ΔP (kPa)", f"{mean_pred_val:.6g}", delta=f"median {median_pred_val:.6g}")

                # Show table with V, best-model pred, ensemble mean
                df_show = pd.DataFrame({
                    'V (m/s)': sub_imp['V'],
                    f'{best_col}': sub_imp[best_col],
                    'Ensemble mean (kPa)': ensemble_mean
                })
                st.subheader("Best-model predictions vs ensemble mean")
                st.dataframe(df_show.reset_index(drop=True), height=300)

                # Plots: overlay model predictions and ensemble
                fig = go.Figure()
                # ensemble
                fig.add_trace(go.Scatter(x=sub_imp['V'], y=ensemble_mean, mode='lines+markers', name='Ensemble mean', line=dict(width=3, dash='dash')))
                # all models
                colors = px.colors.qualitative.Plotly
                i = 0
                for col in model_cols:
                    fig.add_trace(go.Scatter(x=sub_imp['V'], y=sub_imp[col], mode='markers+lines', name=col.replace("dP_","").replace("_kPa",""),
                                             line=dict(color=colors[i % len(colors)], width=1), opacity=0.8))
                    i += 1
                # highlight best
                fig.add_trace(go.Scatter(x=sub_imp['V'], y=sub_imp[best_col], mode='lines+markers', name='Best model (highlight)', line=dict(color='black', width=4)))
                fig.update_layout(title="Model ΔP predictions (improved) vs Ensemble", xaxis_title="Velocity (m/s)", yaxis_title="ΔP (kPa)", legend=dict(orientation='h'))
                st.plotly_chart(fig, use_container_width=True)

                # Bar chart of mse_vs_ensemble
                mse_df = pd.DataFrame(sorted([(k.replace("dP_","").replace("_kPa",""), v) for k,v in mse_vs_ensemble.items() if v is not None], key=lambda x: x[1]), columns=['model','mse'])
                barfig = px.bar(mse_df, x='model', y='mse', title="MSE vs Ensemble (improved set)", labels={'mse':'MSE (kPa^2)'})
                st.plotly_chart(barfig, use_container_width=True)

                # Download button for best-model predictions
                csv_bytes = df_show.to_csv(index=False).encode('utf-8')
                st.download_button("Download best-model predictions (CSV)", data=csv_bytes, file_name=f"improved_best_model_{best_col}.csv", mime='text/csv')
    



else:
    st.write("Configure inputs on the sidebar and click 'Run (process file)' to execute the pipeline.")
# ------------------ Credits (highlight) ------------------
    st.markdown(
    """
    <div style="margin-top:18px; padding:12px; border-radius:10px; background: linear-gradient(90deg,#fffaf0,#f0fbff); box-shadow: 0 2px 6px rgba(0,0,0,0.06); font-size:16px;">
      <strong style="font-size:18px;">Credits:</strong> Made under supervision and support of
      <span style="font-size:18px; font-weight:800; color:#b22222; background: rgba(255,230,230,0.6); padding:4px 8px; border-radius:6px;">Dr. Geetanjali Chauhan</span>
    </div>
    """,
    unsafe_allow_html=True
)
