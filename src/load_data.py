import pandas as pd

column_names = [
    "Lever position (lp)", "Ship speed (v)", "GT shaft torque (GTT)", "GT rate of revolutions (GTn)",
    "Gas Generator rate of revolutions (GGn)", "Starboard Propeller Torque (Ts)", "Port Propeller Torque (Tp)",
    "HP Turbine exit temperature (T48)", "GT Compressor inlet air temperature (T1)", "GT Compressor outlet air temperature (T2)",
    "HP Turbine exit pressure (P48)", "GT Compressor inlet air pressure (P1)", "GT Compressor outlet air pressure (P2)",
    "GT exhaust gas pressure (Pexh)", "Turbine Injection Control (TIC)", "Fuel flow (mf)",
    "GT Compressor decay state coefficient", "GT Turbine decay state coefficient"
]

df = pd.read_csv('/home/natflaz/Documents/IUTinfo/s4/data/ShipIA/UCI CBM Dataset/data.txt', sep='\s+', names=column_names, header=None, engine='python')
df.drop_duplicates(inplace=True)


