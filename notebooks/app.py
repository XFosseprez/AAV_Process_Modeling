import streamlit as st
import os
import numpy as np
import pandas as pd
from scipy.integrate import odeint
import duckdb
import joblib
from pathlib import Path
from scipy.integrate import trapezoid
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import plotly.graph_objects as go
import matplotlib.pyplot as plt

if 'count' not in st.session_state:
    st.session_state.count = 0
if "list_batch" not in st.session_state:
    st.session_state.list_batch = ("No batch produced yet")
    list_batch = st.session_state.list_batch
    
        
st.title("Adeno-Associated Virus production data generation and analysis")
        
def bioprocess_model(y, t, params):
    #dependent variables definition
    X, G, L, A, P = y
    #parameters definition (max growth rate, reaction rate constant (Ki_A (Ammonia), Ki_L (Lactate)), conversion rate)
    mu_max, Ks, Ki_A, Ki_L, Y_xg, Y_lg, Y_ag, q_p = params
    
    #equations for growth, consumption and production
    mu = mu_max * (G / (Ks + G)) * (Ki_A / (Ki_A + A)) * (Ki_L / (Ki_L + L))
    
    dXdt = mu * X
    dGdt = -(1/Y_xg) * mu * X
    dLdt = Y_lg * (1/Y_xg) * mu * X
    dAdt = Y_ag * (1/Y_xg) * mu * X
   # Hybrid Production Logic
    if t > 72:
        if t > 230:
            # Final Plateau: Efficiency drops as growth (mu) slows down
            # This forces the S-curve shape at the very end
            dPdt = q_p * X * (mu / mu_max)   
        else:
            dPdt = q_p * X  # Peak Production phase (Linear/Aggressive)
    else:
        dPdt = 0  # Lag/Growth phase (No production)
        
    return [dXdt, dGdt, dLdt, dAdt, dPdt]

def generate_sabotaged_data(n_iteration):
    all_telemetry = []
    all_outcomes = []
    t_eval = np.linspace(0, 240, 241) 
    
    #defining batches
    batch_id = f"BATCH_{n_iteration:03d}"
    status = "Golden"

    #default values for parameters and initial value for variables
    # mu_max, Ks, Ki_A, Ki_L, Y_xg, Y_lg, Y_ag, q_p
    params = [0.04, 0.5, 15.0, 40.0, 0.4, 0.6, 0.1, 0.0025]        
    y0 = [0.5, 50.0, 0.0, 0.0, 0.0]

    #ouptup value between 0.0 and 1.0
    dice_roll = np.random.random()

    # as the dice roll should have equal probability for all results,
    # there is a 15 % chance to start with lower glucose
    if dice_roll < 0.15: 
        status = "OOS_Glucose_Fail"
        y0[1] = 30.0 

    # 15 % chance to have a toxic drift, 
    # Yag of 0.4 instead of 0.1 means that for the same amount of biomass
    # there is 1.8 times more ammonia produced
    elif dice_roll < 0.30: 
        status = "OOS_Toxic_Drift"
        params[5] = 0.18
            
    # 10 % chance of transfection fail,
    # low transfection is translated by low productivity rate (30 % lower)
    elif dice_roll < 0.40: 
        status = "OOS_Transfection_Fail"
        params[6] = 0.035 
            
    #solving equation system    
    sol = odeint(bioprocess_model, y0, t_eval, args=(params,))

    #creating random normaly distributed noise for all data
    noise = np.random.normal(0, 0.015, sol.shape)# random noise centered at 0 with 1.5 % standard deviation populated accross the shape of the sol matrix
    sol_noisy = sol + (sol * noise) #adding the noise fitted to the actual data with the data
        
    batch_df = pd.DataFrame(sol_noisy, columns=['VCD', 'Glucose', 'Lactate', 'Ammonia', 'Product'])

    base_ph = 7.2
    ph_drop = 0.05 * batch_df['Lactate']
    ph_buffer = 0.01 * batch_df['Ammonia']
    batch_df['pH'] = base_ph - ph_drop + ph_buffer + np.random.normal(0, 0.01, len(t_eval))
        
    batch_df['Hour'] = t_eval
    batch_df['Batch_ID'] = batch_id
    all_telemetry.append(batch_df)

    base_eff = 0.35
    if status == "OOS_Toxic_Drift":
        eff = base_eff * 0.6
    elif status == "OOS_Transfection_Fail":
        eff = base_eff * 0.3
    else:
        eff = base_eff * np.random.normal(1, 0.05)

    total_titer = sol[-1, 4] #fetching the last value for the 5th variable
    full_titer = total_titer * eff
    pct_full = eff * 100

    all_outcomes.append({
        'Batch_ID': batch_id,
            'Total_Titer': total_titer,
            'Full_Titer': full_titer,
            'Percent_Full': pct_full
            #,'Status': status (deactivating status for application)
        })
        
    return pd.concat(all_telemetry), pd.DataFrame(all_outcomes)

def generate_dirty_data(telemetry_df):
    dirty_rows = []
    
    for batch_id in telemetry_df['Batch_ID'].unique():
        batch_data = telemetry_df[telemetry_df['Batch_ID'] == batch_id]
        
        # 1. High Frequency: Ammonia & Lactate (Every 2 hours)
        # We simulate probe data with jittery timestamps
        ammonia_data = batch_data.iloc[::2][['Hour', 'pH', 'Ammonia', 'Lactate', 'Batch_ID']].copy()
        ammonia_data['Hour'] += np.random.normal(0, 0.1, len(ammonia_data)) 
        
        # 2. Low Frequency: VCD & Glucose (Every 24 hours)
        # Mimics a technician coming in once a day
        vcd_data = batch_data.iloc[::24][['Hour', 'VCD', 'Glucose', 'Product', 'Batch_ID']].copy()
        vcd_data['Hour'] += np.random.normal(0, 0.5, len(vcd_data)) 
        
        # Combine into a "messy" long-form table
        dirty_rows.append(ammonia_data)
        dirty_rows.append(vcd_data)
        
    return pd.concat(dirty_rows).sort_values(['Batch_ID', 'Hour'])

def align_telemetry(group):
    # Define a clean 1-hour grid
    clean_grid = np.arange(0, 241, 1)
    
    # We isolate the numeric columns so 'Batch_ID' doesn't cause a warning
    cols_to_sync = ['VCD', 'Glucose', 'Lactate', 'Ammonia', 'Product', 'pH']
    
    # 1. Combine existing hours with the target grid
    target_indices = pd.Index(clean_grid, name='Hour')
    combined_indices = group.set_index('Hour').index.union(target_indices)
    
    # 2. Reindex to the combined set (preserves all original data points)
    return (group.set_index('Hour')[cols_to_sync]
                 .reindex(combined_indices)
                 .sort_index()
                 .interpolate(method='linear') # Interpolates using the float distances
                 .ffill()
                 .bfill()
                 .loc[clean_grid]) # Finally, keep only the exact hours from your grid

@st.cache_resource
def get_db_connection():
    return duckdb.connect()

def generate_data(): 
    con = get_db_connection()
    
    df_telemetry, df_outcomes = generate_sabotaged_data(st.session_state.count)
    st.write(st.session_state.count)
    st.session_state.count += 1
    df_messy = generate_dirty_data(df_telemetry)
    df_cleaned = (
        df_messy
        .groupby('Batch_ID')
        .apply(align_telemetry, include_groups=False)
        .reset_index()
                 )
    # Calculate instantaneous rates in the aligned dataframe
    df_cleaned = df_cleaned.sort_values(['Batch_ID', 'Hour'])
    df_cleaned['dt'] = df_cleaned.groupby('Batch_ID')['Hour'].diff()
    
    # Growth Rate (mu): (ΔVCD / VCD) / Δt
    df_cleaned['Growth_Rate'] = df_cleaned.groupby('Batch_ID')['VCD'].diff() / (df_cleaned['VCD'] * df_cleaned['dt'])
        
    # Specific Production Rate (qp): ΔP / (VCD * Δt)
    df_cleaned['Production_Rate'] = df_cleaned.groupby('Batch_ID')['Product'].diff() / (df_cleaned['VCD'] * df_cleaned['dt'])
        
    # Specific Consumption Rate (qg): ΔG / (VCD * Δt)
    df_cleaned['Consumption_Rate'] = df_cleaned.groupby('Batch_ID')['Glucose'].diff() / (df_cleaned['VCD'] * df_cleaned['dt'])
    df_cleaned = df_cleaned.drop("dt",axis=1)
    
    con.execute("DROP VIEW IF EXISTS temp.telemetry_aligned")
    con.execute("""CREATE TABLE IF NOT EXISTS telemetry_aligned(
            Batch_ID VARCHAR, 
            Hour DOUBLE,
            VCD DOUBLE, 
            Glucose DOUBLE, 
            Lactate DOUBLE, 
            Ammonia DOUBLE, 
            Product DOUBLE,
            pH DOUBLE,
            Growth_Rate DOUBLE,
            Production_Rate DOUBLE,
            Consumption_Rate DOUBLE)
            """)
    
    con.execute("DROP VIEW IF EXISTS temp.outcomes")    
    con.execute("""CREATE TABLE IF NOT EXISTS outcomes(
            Batch_ID VARCHAR, 
            Total_Titer DOUBLE, 
            Full_Titer DOUBLE, 
            Percent_Full DOUBLE)
            """)
    
    df_cleaned = df_cleaned.astype({col: 'object' for col in df_cleaned.columns if df_cleaned[col].dtype.name == 'str'})
    df_outcomes = df_outcomes.astype({col: 'object' for col in df_outcomes.columns if df_outcomes[col].dtype.name == 'str'})

    con.execute("INSERT INTO telemetry_aligned BY NAME SELECT * FROM df_cleaned")
    con.execute("INSERT INTO outcomes BY NAME SELECT * FROM df_outcomes")

    query = """
    SELECT *
    FROM telemetry_aligned 
    """

    df_batches = con.execute(query).df()
    
    step_1.write(df_batches.tail())
    list_batches = df_batches["Batch_ID"].unique()
    return list_batches

## First Menu: Data generation
with st.expander("Step 1: generating AAV culture data") as step_1:
        if st.button(
            label="Generate data",
            key="generate_data_btn",
            help="Produces files for three moments of the culture: 75h, 100h and the end of the culture"
        ):
            st.session_state.list_batch = generate_data()
    
        list_batch = st.session_state.list_batch

## Preparing for Random forest classifier
def preparing_features(df_tel, df_out, checkpoint_hour):
    df_tel = df_tel.sort_values(by=['Batch_ID', 'Hour']).reset_index(drop=True)
    processed_features = []
    process_variables = ['VCD', 'Glucose', 'Lactate', 'Ammonia', 'Product', 'pH', 'Growth_Rate', 'Production_Rate', 'Consumption_Rate']
    
    for batch_id, batch_group in df_tel.groupby('Batch_ID'):
        # Limit data to given checkpoint hour (function argument) and put in chronological order
        group_to_hour = batch_group[batch_group['Hour'] <= checkpoint_hour].sort_values('Hour')
            
        # Creates a dataframe populated with the current batch_id of the loop
        batch_features = pd.DataFrame({'Batch_ID': [batch_id]})

        # Locate the checkpoint row
        matched_row = group_to_hour[group_to_hour['Hour'] == checkpoint_hour]

        # Extract the state and find its relative integer position in the group
        if not matched_row.empty:
            label = matched_row.index[0]
            checkpoint_index = group_to_hour.index.get_loc(label)

        # Calculating and/or storing current value, lag1 and 2, rolling average and standard deviation
        # cumulative sum
        for col in process_variables:
            batch_features[f'{col}_current'] = group_to_hour[col].iloc[checkpoint_index]
            
            batch_features[f'{col}_lag1'] = group_to_hour[col].iloc[checkpoint_index - 1]
            batch_features[f'{col}_lag2'] = group_to_hour[col].iloc[checkpoint_index - 2]
            
            short_window = group_to_hour[col].iloc[(checkpoint_index - 2):(checkpoint_index + 1)]
            batch_features[f'{col}_roll_mean3'] = short_window.mean()
            batch_features[f'{col}_roll_std3'] = short_window.std()
            
            if checkpoint_index >= 11:
                macro_window = group_to_hour.iloc[(checkpoint_index - 11):(checkpoint_index + 1)]
                hourly_rates = macro_window[col].diff().dropna() / macro_window['Hour'].diff().dropna()
                batch_features[f'{col}_avg_rate_12h'] = hourly_rates.mean()
            else:
                batch_features[f'{col}_avg_rate_12h'] = np.nan
                
            y = group_to_hour[col].values
            x = group_to_hour['Hour'].values
            if len(x) > 1:
                batch_features[f'{col}_cum_area'] = trapezoid(y, x)
            else:
                batch_features[f'{col}_cum_area'] = 0.0
                
        processed_features.append(batch_features)
              
    df_features_matrix = pd.concat(processed_features, ignore_index=True)
    df_modeling_set = pd.merge(df_features_matrix, df_out[['Batch_ID', 'Full_Titer']], on='Batch_ID', how='inner')

    
    return df_modeling_set

## Second Menu: Predictive classification for Full titer
    
@st.cache_data
def classifying_batch (batch, time):
    con = get_db_connection()
    
    query = """
    SELECT *
    FROM telemetry_aligned 
    WHERE "Batch_ID" = ?
    """
    
    telemetry_random_forest = con.execute(query, [batch]).df()
    
    query_2 = """
    SELECT 
        o.Batch_ID, 
        o.Full_Titer, 
    FROM outcomes o
    WHERE "Batch_ID" = ?
    """
    
    outcome_random_forest = con.execute(query_2, [batch]).df()
    
    df_checkpoint = preparing_features(telemetry_random_forest, outcome_random_forest, time)
    X = df_checkpoint.drop(columns=["Full_Titer", "Batch_ID"]).copy()
    
    if time == 100:
        classifier_100h_model_path = Path("C:\dev\AAV_Process_Modeling\models") / "Classifier_100h_model.joblib"
        classifier_100h_model = joblib.load(classifier_100h_model_path)
        y_hat = classifier_100h_model.predict(X)
    else:
        classifier_75h_model_path = Path("C:\dev\AAV_Process_Modeling\models") / "Classifier_100h_model.joblib"
        classifier_75h_model = joblib.load(classifier_75h_model_path)
        y_hat = classifier_75h_model.predict(X)
    return y_hat

with st.expander("Step 2: predictive classification for Full titer") as step_2:
    selected_batch_1 = st.select_slider(label= "Batch selection", key="batch_select_rf", options= list_batch )
    selected_time = st.select_slider(label= "Timepoint to evaluate Full Titer (h)", options=(75, 100), value= 75, width=100)
    if selected_batch_1 == "No batch produced yet":
        st.write("No batch available to analyse")
    else:
        full_titer_prediction = np.round(classifying_batch (selected_batch_1, selected_time), 2)
        st.write(f"The predicted final full titer based on data at {selected_time} hours is {full_titer_prediction} g/L")
    
@st.cache_data
def golden_batch(batch):
    con = get_db_connection()

    query_3 = """
    SELECT 
        o.Batch_ID, 
        o.Full_Titer, 
        MAX(t.Ammonia) AS Max_Ammonia,
        MAX(t.Lactate) AS Max_Lactate,
        MIN(t.pH) AS Min_pH,
        AVG(CASE WHEN t.Hour >= 0 THEN t.Growth_Rate END) AS Avg_Growth_Rate,
        AVG(CASE WHEN t.Hour >= 72 THEN t.Production_Rate END) AS Avg_Prod_Rate,
        AVG(CASE WHEN t.Hour >= 0 THEN t.Consumption_Rate END) AS Avg_Gluc_Cons
    FROM outcomes o
    JOIN telemetry_aligned t ON o.Batch_ID = t.Batch_ID
    WHERE o.Batch_ID = ?
    GROUP BY o.Batch_ID, o.Full_Titer
    """
    df_pca = con.execute(query_3, [batch]).df()
       
    feature_cols = ['Max_Ammonia', 'Max_Lactate', 'Min_pH', 
                    'Avg_Growth_Rate', 'Avg_Prod_Rate', 'Avg_Gluc_Cons']

    query_4 = """
    SELECT 
        *
    FROM outcomes o
    WHERE o.Batch_ID = ?
    """
    
    df_test = con.execute(query_4, [batch]).df()
    st.write(df_test)
    df_pca = df_pca.dropna(subset=feature_cols)
    golden_batch_model_path = Path("C:\dev\AAV_Process_Modeling\models") / "gold_batch.joblib"
    golden_batch_pipe = joblib.load(golden_batch_model_path)
    principal_components = golden_batch_pipe.transform(df_pca[feature_cols])
    df_plot = pd.DataFrame(data=principal_components, columns=['PC1', 'PC2', 'PC3'])
    
    #code for plotting the new batch within the golden PCA space
    eigenvalues_threshold = (2.06079346, 1.36070661, 1.28002434)
    t2_threshold = 14.214428434604127
    radius_x = np.sqrt(t2_threshold * eigenvalues_threshold[0])
    radius_y = np.sqrt(t2_threshold * eigenvalues_threshold[1])
    radius_z = np.sqrt(t2_threshold * eigenvalues_threshold[2])
    
    # 3. Create the Ellipse Shape
    phi = np.linspace(0, np.pi, 50)
    theta = np.linspace(0, 2 * np.pi, 50)
    phi, theta = np.meshgrid(phi, theta)
    
    x_ellipse = radius_x * np.sin(phi) * np.cos(theta)
    y_ellipse = radius_y * np.sin(phi) * np.sin(theta)
    z_ellipse = radius_z * np.cos(phi)
    
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter3d(
        x=df_plot.loc[:,'PC1'],
        y=df_plot.loc[:,'PC2'],
        z=df_plot.loc[:,'PC3'],
        mode='markers',
        marker=dict(size=4),
        name=f'Batch: {batch}'
        ))
    
    fig.add_trace(go.Surface(
        x=x_ellipse, y=y_ellipse, z=z_ellipse,
        opacity=0.2,
        showscale=False,
        colorscale=[[0, 'gray'], [1, 'gray']],
        name='95% T2 Ellipsoid'
    ))
    
    fig.update_layout(
        scene=dict(
            xaxis_title='PC1',
            yaxis_title='PC2',
            zaxis_title='PC3'
        ),
        title="3D PCA Score Plot with Hotelling T2 Ellipsoid"
    )
    
    fig.update_layout(
        width=1200,
        height=800,
        scene=dict(
            xaxis_title='PC1',
            yaxis_title='PC2',
            zaxis_title='PC3'
        ),
    )
    st.plotly_chart(fig, use_container_width=True, height=500)

    eigenvalues = golden_batch_pipe["pca"].explained_variance_
    t2_value = np.sum((principal_components**2) / eigenvalues, axis=1)
    st.write(f"T2 value is {np.round(t2_value,2)} compared to the threshold {np.round(t2_threshold,2)}")

    all_scaled = golden_batch_pipe["scaler"].transform(df_pca[feature_cols])
    reconstructed = golden_batch_pipe["pca"].inverse_transform(principal_components)
    residuals = all_scaled - reconstructed
    spe_value = np.sum(residuals**2, axis=1)
    spe_threshold = 5.5855878193436315
    st.write(f"SPE value is {np.round(spe_value,2)} compared to the threshold {np.round(spe_threshold,2)}")

    contributions = np.zeros(len(feature_cols))
    for i in range(len(feature_cols)):
        contribution_i = 0
        for pc_idx in range(golden_batch_pipe["pca"].n_components_):
            loading = golden_batch_pipe["pca"].components_[pc_idx, i]
            score = principal_components[0, pc_idx]
            eigenvalue = golden_batch_pipe["pca"].explained_variance_[pc_idx]
            contribution_i += score * loading / eigenvalue
        contributions[i] =  all_scaled[0,i] * contribution_i

    fig_2, ax = plt.subplots()
    ax.bar(feature_cols, contributions, color='crimson', edgecolor='black', alpha=0.8)
    ax.axhline(0, color='grey', lw=1)
    ax.set_title(f'T² Variable Contribution Plot for Batch: {batch}')
    ax.set_ylabel('Contribution Weight')
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
    ax.grid(axis='y', linestyle=':', alpha=0.6)
    fig_2.tight_layout()
    st.pyplot(fig_2, use_container_width=True)
                    
with st.expander("Step 3: Batch comparison with golden batch profile") as step_3:
    selected_batch_2 = st.select_slider(label= "Batch selection", key="batch_select_pca", options= list_batch )
    if selected_batch_2 == "No batch produced yet":
        st.write("No batch available to analyse")
    else:
        golden_batch(selected_batch_2)
       