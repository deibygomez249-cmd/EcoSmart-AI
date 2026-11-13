from flask import Flask, render_template, request, flash, redirect, url_for
from flask_login import LoginManager, login_user, login_required, logout_user, current_user
import os
from database import db, User, Device
from models.ecosmart import (
    train_all_models, get_dashboard_charts, predict_single, 
    model_metadata, trained_models, load_data, get_anomaly_plot,
    get_clustering_plot, get_metrics_chart, get_zone_data,
    get_data_explorer_info, get_anomalies_data, get_insights_data
)

app = Flask(__name__)
app.secret_key = 'ecosmart_secret'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///ecosmart.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db.init_app(app)

login_manager = LoginManager()
login_manager.login_view = 'login'
login_manager.init_app(app)

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

# --- RUTAS DE AUTENTICACIÓN ---

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        
        user = User.query.filter_by(username=username).first()
        if user:
            flash('Username already exists.', 'danger')
        else:
            new_user = User(username=username)
            new_user.set_password(password)
            db.session.add(new_user)
            db.session.commit()
            flash('Registration successful! Please log in.', 'success')
            return redirect(url_for('login'))
    return render_template('auth/register.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        user = User.query.filter_by(username=username).first()
        
        if user and user.check_password(password):
            login_user(user)
            return redirect(url_for('ecosmart_concepts'))
        else:
            flash('Invalid username or password.', 'danger')
    return render_template('auth/login.html')

@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('login'))

# --- RUTAS PRINCIPALES (Protegidas) ---

@app.route('/')
def home():
    return redirect(url_for('ecosmart_concepts'))

@app.route('/ecosmart/concepts')
@login_required
def ecosmart_concepts():
    try:
        charts = get_dashboard_charts()
        zones = get_zone_data()
        return render_template('ecosmart/concepts.html', data=charts, zones=zones)
    except Exception as e:
        return render_template('ecosmart/concepts.html', error=str(e))

@app.route('/ecosmart/data-explorer')
@login_required
def ecosmart_data_explorer():
    try:
        info = get_data_explorer_info()
        return render_template('ecosmart/data_explorer.html', info=info)
    except Exception as e:
        return render_template('ecosmart/data_explorer.html', error=str(e))

@app.route('/ecosmart/application', methods=['GET', 'POST'])
@login_required
def ecosmart_application():
    prediction_result = None
    error_msg = None
    
    if request.method == 'POST':
        action = request.form.get('action')
        
        if action == 'train':
            try:
                train_all_models()
                flash('All models trained successfully!', 'success')
            except Exception as e:
                flash(f'Error: {str(e)}', 'danger')
            return redirect(url_for('ecosmart_application'))
            
        elif action == 'predict':
            try:
                features = {
                    'T1': float(request.form.get('T1', 20.0)),
                    'RH_1': float(request.form.get('RH_1', 40.0)),
                    'T_out': float(request.form.get('T_out', 10.0)),
                    'Windspeed': float(request.form.get('Windspeed', 2.0)),
                }
                _, X_train, _, _, _ = load_data()
                for col in X_train.columns:
                    if col not in features:
                        features[col] = float(X_train[col].mean())
                prediction_result = predict_single(features)
            except Exception as e:
                error_msg = f"Error: {str(e)}"
                
    clustering_plot = None
    if trained_models:
        try: clustering_plot = get_clustering_plot()
        except: pass

    return render_template('ecosmart/application.html', 
                           metadata=model_metadata, 
                           has_models=bool(trained_models),
                           prediction=prediction_result,
                           error=error_msg,
                           clustering_plot=clustering_plot)

@app.route('/ecosmart/metrics')
@login_required
def ecosmart_metrics():
    anomaly_plot = None
    metrics_plot = None
    if trained_models:
        try:
            anomaly_plot = get_anomaly_plot()
            metrics_plot = get_metrics_chart()
        except:
            pass
    return render_template('ecosmart/metrics.html', 
                           metadata=model_metadata, 
                           anomaly_plot=anomaly_plot,
                           metrics_plot=metrics_plot)

@app.route('/ecosmart/anomalies', methods=['GET', 'POST'])
@login_required
def ecosmart_anomalies():
    contamination = 0.05
    if request.method == 'POST':
        val_str = request.form.get('contamination', '0.05').replace(',', '.')
        try:
            contamination = float(val_str)
        except ValueError:
            contamination = 0.05
    
    try:
        data = get_anomalies_data(contamination)
        return render_template('ecosmart/anomalies.html', data=data, contamination=contamination)
    except Exception as e:
        print("ERROR EN ANOMALIAS:", str(e))
        return render_template('ecosmart/anomalies.html', error=str(e))

@app.route('/ecosmart/insights')
@login_required
def ecosmart_insights():
    try:
        data = get_insights_data()
        return render_template('ecosmart/insights.html', data=data)
    except Exception as e:
        return render_template('ecosmart/insights.html', error=str(e))

# --- RUTAS DE DISPOSITIVOS ---

@app.route('/devices')
@login_required
def devices():
    user_devices = Device.query.filter_by(user_id=current_user.id).all()
    
    total_devices = len(user_devices)
    total_consumption = sum(d.consumption for d in user_devices)
    
    # Calcular el porcentaje de consumo de cada dispositivo
    for d in user_devices:
        if total_consumption > 0:
            d.percentage = round((d.consumption / total_consumption) * 100, 1)
        else:
            d.percentage = 0
    
    return render_template('devices.html', 
                           devices=user_devices, 
                           total_devices=total_devices, 
                           total_consumption=round(total_consumption, 2))

@app.route('/devices/add', methods=['POST'])
@login_required
def add_device():
    name = request.form.get('name')
    device_type = request.form.get('device_type')
    location = request.form.get('location')
    
    consumption_str = request.form.get('consumption', '0').replace(',', '.')
    try:
        consumption = float(consumption_str)
    except ValueError:
        consumption = 0.0
    
    if name and device_type and location:
        new_device = Device(
            name=name, 
            device_type=device_type, 
            location=location, 
            consumption=consumption,
            user_id=current_user.id
        )
        db.session.add(new_device)
        db.session.commit()
        flash('Device added successfully!', 'success')
    else:
        flash('Please fill all required fields.', 'danger')
        
    return redirect(url_for('devices'))

if __name__ == '__main__':
    with app.app_context():
        db.create_all()
        
    port = int(os.environ.get('PORT', 5000))
    app.run(debug=True, host='0.0.0.0', port=port)