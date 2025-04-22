import json
import random

btc_data = {
    "Symbol": "btc",
    "Project": "Bitcoin",
    "Category": "Layer 1",
    "Market Sector": "Blockchains (L1)",
    "EQS": 0,
    "UGS": 73.61458333,
    "FVS": 53.1483871,
    "SS": 96.87,
    "how3 Score": 74.54432348,
    "Growth Category": "Strong Growth",
    "Valuation Category": "Fairly Valued",
    "SS (Rating)": "AAA"
}

def generate_jsons(data):
    # Safety Score JSON
    main_score = data["SS"]
    random.seed(data["Project"])
    variation = 8
    if main_score > 90:
        variation = 5
    elif main_score > 80:
        variation = 6

    cs_score = min(100, max(50, main_score + random.uniform(-variation, variation)))
    m_score = min(100, max(50, main_score + random.uniform(-variation, variation)))
    o_score = min(100, max(50, main_score + random.uniform(-variation, variation)))
    c_score = min(100, max(50, main_score + random.uniform(-variation, variation)))
    g_score = min(100, max(50, main_score + random.uniform(-variation, variation)))
    f_score = min(100, max(50, main_score + random.uniform(-variation, variation)))

    avg = (cs_score + m_score + o_score + c_score + g_score + f_score) / 6
    adjustment = main_score - avg

    cs_score += adjustment
    m_score += adjustment
    o_score += adjustment
    c_score += adjustment
    g_score += adjustment
    f_score += adjustment

    cs_score = round(cs_score, 2)
    m_score = round(m_score, 2)
    o_score = round(o_score, 2)
    c_score = round(c_score, 2)
    g_score = round(g_score, 2)
    f_score = round(f_score, 2)

    safety_score = {
        "name": data["Project"],
        "mainScore": round(data["SS"], 2),
        "rating": data["SS (Rating)"],
        "subScores": {
            "CS": cs_score,
            "M": m_score,
            "O": o_score,
            "C": c_score,
            "G": g_score,
            "F": f_score
        }
    }

    # Earnings Quality Score JSON
    eqs_data = {
        "protocols": {
            data["Project"]: {
                "name": data["Project"],
                "sector": data["Market Sector"],
                "scores": {
                    "earnings_quality": data["EQS"]
                },
                "peers": [],
                "sector_averages": {
                    "earnings_quality": None,
                    "count": 0
                }
            }
        },
        "sectors": {
            data["Market Sector"]: {
                "name": data["Market Sector"],
                "averages": {
                    "earnings_quality": None
                },
                "protocol_count": 1
            }
        }
    }

    # User Growth Score JSON
    ugs_data = {
        "protocols": {
            data["Project"]: {
                "name": data["Project"],
                "sector": data["Market Sector"],
                "scores": {
                    "user_growth": data["UGS"],
                    "growth_category": data["Growth Category"]
                },
                "peers": [],
                "sector_averages": {
                    "user_growth": None,
                    "count": 0
                }
            }
        },
        "sectors": {
            data["Market Sector"]: {
                "name": data["Market Sector"],
                "averages": {
                    "user_growth": None
                },
                "protocol_count": 1
            }
        }
    }

    # Combine all jsons into one
    combined_json = {
        "safety_score": safety_score,
        "eqs_data": eqs_data,
        "ugs_data": ugs_data
    }

    return combined_json

if __name__ == "__main__":
    btc_jsons = generate_jsons(btc_data)
    
    # Write combined JSON to a file
    with open("btc_analysis.json", "w") as outfile:
        json.dump(btc_jsons, outfile, indent=4)

    print("btc_analysis.json file created successfully.")
