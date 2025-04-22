import pandas as pd
import json
import random
import os
from io import StringIO  # Import StringIO from io module, not pandas

def generate_safety_scores(csv_data, output_file):
    """
    Generate safety score JSONs for each project based on SS column and rating
    
    Args:
        csv_data: CSV data as a string or DataFrame
        output_file: Where to save the JSON data
    """
    print("Generating safety scores...")
    
    # Load CSV data
    if isinstance(csv_data, str):
        # Load from string (tab-separated)
        df = pd.read_csv(StringIO(csv_data), sep='\t')
    else:
        # Assume it's already a DataFrame
        df = csv_data
    
    print(f"Loaded {len(df)} projects from CSV")
    
    # Create safety scores
    safety_scores = []
    
    for _, row in df.iterrows():
        project_name = row['Project']
        main_score = float(row['SS'])
        rating = row['SS (Rating)']
        
        # Generate realistic sub-scores based on the main score
        # We'll set the random seed based on the project name for consistency
        random.seed(project_name)
        
        # Base variation - how much scores can deviate from main score
        variation = 8
        
        # Stronger projects have less variation
        if main_score > 90:
            variation = 5
        elif main_score > 80:
            variation = 6
        
        # Generate sub-scores that average to approximately the main score
        cs_score = min(100, max(50, main_score + random.uniform(-variation, variation)))
        m_score = min(100, max(50, main_score + random.uniform(-variation, variation)))
        o_score = min(100, max(50, main_score + random.uniform(-variation, variation)))
        c_score = min(100, max(50, main_score + random.uniform(-variation, variation)))
        g_score = min(100, max(50, main_score + random.uniform(-variation, variation)))
        f_score = min(100, max(50, main_score + random.uniform(-variation, variation)))
        
        # Adjust to ensure average is very close to main score
        avg = (cs_score + m_score + o_score + c_score + g_score + f_score) / 6
        adjustment = main_score - avg
        
        cs_score += adjustment
        m_score += adjustment
        o_score += adjustment
        c_score += adjustment
        g_score += adjustment
        f_score += adjustment
        
        # Round to 2 decimal places
        cs_score = round(cs_score, 2)
        m_score = round(m_score, 2)
        o_score = round(o_score, 2)
        c_score = round(c_score, 2)
        g_score = round(g_score, 2)
        f_score = round(f_score, 2)
        
        # Create the score object
        project_data = {
            "name": project_name,
            "mainScore": round(main_score, 2),
            "rating": rating,
            "subScores": {
                "CS": cs_score,  # Code Security
                "M": m_score,    # Market
                "O": o_score,    # Operational
                "C": c_score,    # Community
                "G": g_score,    # Governance
                "F": f_score     # Fundamental
            }
        }
        
        safety_scores.append(project_data)
    
    # Save to JSON file
    with open(output_file, 'w') as f:
        json.dump(safety_scores, f, indent=2)
    
    print(f"Generated safety scores for {len(safety_scores)} projects")
    print(f"Data saved to {output_file}")
    
    return safety_scores

def generate_safety_scores_for_one_project(project_data, output_file=None):
    """
    Generate safety score JSON for a single project
    
    Args:
        project_data: Dictionary with project info
        output_file: Where to save the JSON data (optional)
    
    Returns:
        The generated safety score data
    """
    project_name = project_data["name"]
    main_score = float(project_data["score"])
    rating = project_data["rating"]
    
    # Generate realistic sub-scores based on the main score
    random.seed(project_name)
    
    # Base variation
    variation = 8
    if main_score > 90:
        variation = 5
    elif main_score > 80:
        variation = 6
    
    # Generate sub-scores
    cs_score = min(100, max(50, main_score + random.uniform(-variation, variation)))
    m_score = min(100, max(50, main_score + random.uniform(-variation, variation)))
    o_score = min(100, max(50, main_score + random.uniform(-variation, variation)))
    c_score = min(100, max(50, main_score + random.uniform(-variation, variation)))
    g_score = min(100, max(50, main_score + random.uniform(-variation, variation)))
    f_score = min(100, max(50, main_score + random.uniform(-variation, variation)))
    
    # Adjust to ensure average is close to main score
    avg = (cs_score + m_score + o_score + c_score + g_score + f_score) / 6
    adjustment = main_score - avg
    
    cs_score += adjustment
    m_score += adjustment
    o_score += adjustment
    c_score += adjustment
    g_score += adjustment
    f_score += adjustment
    
    # Round to 2 decimal places
    cs_score = round(cs_score, 2)
    m_score = round(m_score, 2)
    o_score = round(o_score, 2)
    c_score = round(c_score, 2)
    g_score = round(g_score, 2)
    f_score = round(f_score, 2)
    
    # Create the score object
    safety_score = {
        "name": project_name,
        "mainScore": round(main_score, 2),
        "rating": rating,
        "subScores": {
            "CS": cs_score,
            "M": m_score,
            "O": o_score,
            "C": c_score,
            "G": g_score,
            "F": f_score
        }
    }
    
    # Save to file if specified
    if output_file:
        with open(output_file, 'w') as f:
            json.dump(safety_score, f, indent=2)
    
    return safety_score

if __name__ == "__main__":
    import sys
    
    # Check if tab-separated CSV data is provided
    if len(sys.argv) > 1:
        csv_file = sys.argv[1]
        output_file = sys.argv[2] if len(sys.argv) > 2 else "safety_scores.json"
        
        # Load CSV from file
        df = pd.read_csv(csv_file, sep='\t')
        generate_safety_scores(df, output_file)
    else:
        # Use the CSV data from paste.txt
        csv_data = """Symbol	Project	Category	Market Sector	EQS	UGS	FVS	SS	how3 Score	Growth Category	Valuation Category	SS (Rating)
aave	Aave	DeFi		62.67057363	99.07407407	70.5859375	90.35	80.6701463	Exceptional Growth	Undervalued	AA
aero	Aerodrome Finance	DeFi		60.78862235	88.64047174	74.745671	79.36	75.88369127	Exceptional Growth	Undervalued	A
algo	Algorand	Layer 1		38.51705429	57.27272727	52.52403846	87.49	58.950955	Steady Growth	Fairly Valued	AA
apt	Aptos	Layer 1		43.49478334	74.57650162	55.43990385	94.28	66.9477972	Strong Growth	Fairly Valued	AAA
arb	Arbitrum	Layer 2		64.95491371	96.30346291	70.41071429	94.3	81.49227273	Exceptional Growth	Undervalued	AAA
avax	Avalanche	Layer 1		39.96282984	63.59943978	61.80288462	94.12	64.87128856	Steady Growth	Fairly Valued	AAA
sAVAX	BENQI Liquid Staked AVAX	DeFi		40.20428212	70.42702242	60.671875	79.85	62.78829488	Strong Growth	Fairly Valued	A
bnb	BNB Chain	Layer 1		48.50092318	94.40882035	67.66666667	94.69	76.31660255	Exceptional Growth	Fairly Valued	AAA
celo	Celo	Layer 1		35.30555657	56.73683261	72.05833333	78.76	60.71518063	Steady Growth	Undervalued	BBB
link	Chainlink	Infrastructure		26.85214894	85.72299652	30.9375	92.06	58.89316136	Exceptional Growth	Overvalued	AA
comp	Compound	DeFi		32.43084977	79.99756335	42.6953125	91.56	61.6709314	Strong Growth	Fairly Valued	AA
cvx	Convex Finance	DeFi		52.47063715	94.95356794	62.76785714	85.76	73.98801556	Exceptional Growth	Fairly Valued	A
atom	Cosmos	Layer 1		26.15789224	55	36.34943182	94.25	52.93933101	Steady Growth	Overvalued	AAA
crv	Curve DAO Token	DeFi		56.84522302	80.46152922	50.34801136	89.29	69.2361909	Exceptional Growth	Fairly Valued	AA
ngl	Entangle	Infrastructure		28.18419007	49.52380952	41.80803571	73.67	48.29650883	Steady Growth	Fairly Valued	BBB
ena	Ethena DeFi	DeFi		57.15610751	80.86119468	76.46969697	90.55	76.25924979	Exceptional Growth	Undervalued	AA
ena	Ethena	Stablecoin 		58.228517	85.625	65.07440476	90.55	74.86948044	Exceptional Growth	Fairly Valued	AA
eth	Ethereum	Layer 1		26.40262319	80.74585137	62.93269231	96.23	66.57779172	Exceptional Growth	Fairly Valued	AAA
fil	Filecoin	Infrastructure		22.23684806	64.09090909	29.43181818	86.39	50.53739383	Steady Growth	Overvalued	A
gmx	GMX	DeFi		50.31912787	95.64567206	75.34722222	91.92	78.30800554	Exceptional Growth	Undervalued	AA
gzro	Gravity	Layer 1		33.303054	81.36211567	71.04395604	85.13	67.70978143	Exceptional Growth	Undervalued	A
imx	Immutable X	Layer 2		62.73696332	90.02038226	92.61538462	90.06	83.85818255	Exceptional Growth	Undervalued	AA
inj	Injective	Layer 1		49.72054029	64.14820474	81.10576923	83.9	69.71862856	Steady Growth	Undervalued	A
icp	Internet Computer	Layer 1		34.99216143	74.21487603	91.19318182	88.3	72.17505482	Strong Growth	Undervalued	AA
jto	Jito Labs	Infrastructure		40.57981265	92.94117647	52.91666667	82.52	67.23941395	Exceptional Growth	Fairly Valued	A
ldo	Lido DAO	DeFi		62.60100053	79.24632353	70.40625	92.06	76.07839352	Strong Growth	Undervalued	AA
mpl	Maple Finance	DeFi		54.32192715	63.62573099	56.83035714	92.39	66.79200382	Steady Growth	Fairly Valued	AA
moca	Mocaverse	GameFi		27.03023138	40.23071721	29.1311553	85.15	45.38552597	Slow Growth	Overvalued	A
egld	MultiversX	Layer 1		30.99086042	52.53787879	53.08238636	83.43	55.01028139	Steady Growth	Fairly Valued	A
near	NEAR Protocol	Layer 1		45.60837441	34.00568182	49.32692308	90.96	54.97524483	Slow Growth	Fairly Valued	AA
trac	OriginTrail	AI		41.77057313	71.47321429	54.12202381	70.57	59.48395281	Strong Growth	Fairly Valued	BB
cake	PancakeSwap	DeFi		20.61052856	99.14772727	81.26623377	91.84	73.2161224	Exceptional Growth	Undervalued	AA
pendle	Pendle	DeFi		44.2029527	94.40474858	51.42857143	89.37	69.10367041	Exceptional Growth	Fairly Valued	AA
dot	Polkadot	Layer 1		29.47191562	43.33908279	36.82692308	84.11	48.43698037	Slow Growth	Overvalued	A
red	RedStone	Layer 1		32.06323209	68.96590909	60.94951923	84.93	61.7271651	Strong Growth	Fairly Valued	A
ron	Ronin Network	Layer 1		19.76231383	54.89071386	44.23076923	75.52	48.60094923	Steady Growth	Fairly Valued	BBB
mkr	Sky (formerly MakerDAO)	DeFi		77.66717859	63.81696429	45	91.99	69.61853572	Steady Growth	Fairly Valued	AA
sol	Solana	Layer 1		36.12038313	96.22273152	79.43181818	91.78	75.88873321	Exceptional Growth	Undervalued	AA
s	Sonic Labs (prev. Fantom)	Layer 1		21.67443534	32.89152024	36.875	89.26	45.1752389	Slow Growth	Overvalued	AA
ethx	Stader ETHx	DeFi		39.25070818	44.74264706	11.25	64.22	39.86583881	Slow Growth	Overvalued	BB
sushi	Sushiswap	DeFi		47.83216669	88.87737216	49.53125	85.11	67.83769721	Exceptional Growth	Fairly Valued	A
snx	Synthetix	DeFi		30.53611204	80.07145256	49.49404762	83.71	51.42857143	Exceptional Growth	Fairly Valued	A
trx	TRON	Layer 1		89.21328539	88.24044012	93.07692308	87.49	89.50516215	Exceptional Growth	Undervalued	AA
vet	Vechain	DeFi		62.67057363	99.07407407	70.5859375	92.59	81.2301463	Exceptional Growth	Undervalued	AA
vUSDT	Venus USDT	DeFi		58.72937	86.17202729	84.21875	68.62	74.43503682	Exceptional Growth	Undervalued	BB
zk	zkSync	Layer 2		37.32803842	76.67636132	37.21840659	85.15	59.09320158	Strong Growth	Overvalued	A"""
        
        generate_safety_scores(csv_data, "safety_scores.json")
        
        # Also generate individual files for each project
        df = pd.read_csv(StringIO(csv_data), sep='\t')
        os.makedirs("individual_scores", exist_ok=True)
        
        for _, row in df.iterrows():
            project_name = row['Project']
            file_name = f"individual_scores/{project_name.lower().replace(' ', '_')}.json"
            
            project_data = {
                "name": project_name,
                "score": float(row['SS']),
                "rating": row['SS (Rating)']
            }
            
            generate_safety_scores_for_one_project(project_data, file_name)
            print(f"Generated file for {project_name}")