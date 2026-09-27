"""
Smart Bin Recommendation Rules & Waste Disposal Guidelines.
Maps each predicted waste category to its recommended disposal bin, icon,
actionable disposal instruction, environmental tip, and UI color theme.
"""

from typing import Dict, Any, List

# Comprehensive mapping of waste categories to disposal rules
WASTE_RULES: Dict[str, Dict[str, Any]] = {
    "Plastic": {
        "category": "Plastic",
        "recommended_bin": "♻️ Plastic Bin",
        "bin_short": "Plastic Bin",
        "icon": "♻️",
        "bin_color": "#3b82f6",       # Vibrant Blue
        "badge_bg": "rgba(59, 130, 246, 0.15)",
        "instruction": "Place plastic waste in the plastic recycling bin. Empty liquids and rinse bottles or containers before disposal.",
        "eco_tip": "Recycling one plastic bottle saves enough energy to power an LED lamp for over 24 hours.",
        "recyclable": True,
        "examples": ["Water bottles", "Soda bottles", "Plastic containers", "Shampoo bottles", "Rigid packaging"],
    },
    "Paper": {
        "category": "Paper",
        "recommended_bin": "📄 Paper Bin",
        "bin_short": "Paper Bin",
        "icon": "📄",
        "bin_color": "#0ea5e9",       # Sky Blue
        "badge_bg": "rgba(14, 165, 233, 0.15)",
        "instruction": "Place clean, dry paper or flattened cardboard into the paper recycling bin. Avoid greasy or wet paper.",
        "eco_tip": "Recycling 1 ton of paper saves 17 mature trees and 7,000 gallons of water.",
        "recyclable": True,
        "examples": ["Newspapers", "Cardboard boxes", "Notebooks", "Office paper", "Paper bags"],
    },
    "Metal": {
        "category": "Metal",
        "recommended_bin": "🥫 Metal Bin",
        "bin_short": "Metal Bin",
        "icon": "🥫",
        "bin_color": "#f59e0b",       # Amber Gold
        "badge_bg": "rgba(245, 158, 11, 0.15)",
        "instruction": "Empty all liquids and place aluminum cans, tin cans, or clean metal foil in the metal recycling bin.",
        "eco_tip": "Aluminum and steel can be recycled infinitely with up to 95% less energy than producing raw metal.",
        "recyclable": True,
        "examples": ["Beverage cans", "Food tins", "Aluminum foil", "Metal bottle caps", "Steel containers"],
    },
    "Glass": {
        "category": "Glass",
        "recommended_bin": "🍾 Glass Bin",
        "bin_short": "Glass Bin",
        "icon": "🍾",
        "bin_color": "#06b6d4",       # Cyan / Teal
        "badge_bg": "rgba(6, 182, 212, 0.15)",
        "instruction": "Handle carefully and place unbroken glass bottles or jars in the glass recycling bin after removing lids.",
        "eco_tip": "Glass is 100% recyclable and can be endlessly remelted without any loss in purity or quality.",
        "recyclable": True,
        "examples": ["Glass beverage bottles", "Jam & sauce jars", "Glass flasks", "Condiment bottles"],
    },
    "Organic": {
        "category": "Organic",
        "recommended_bin": "🌱 Organic/Wet Waste Bin",
        "bin_short": "Organic/Wet Waste Bin",
        "icon": "🌱",
        "bin_color": "#10b981",       # Emerald Green
        "badge_bg": "rgba(16, 185, 129, 0.15)",
        "instruction": "Place food scraps, fruit peels, and biodegradable waste in the organic/wet waste composting bin.",
        "eco_tip": "Composting organic waste prevents landfill methane emissions and creates nutrient-rich soil.",
        "recyclable": True,
        "examples": ["Fruit peels (banana, orange)", "Vegetable scraps", "Leftover food", "Tea bags & coffee grounds", "Leaves"],
    },
    "General Waste": {
        "category": "General Waste",
        "recommended_bin": "🗑️ General Waste Bin",
        "bin_short": "General Waste Bin",
        "icon": "🗑️",
        "bin_color": "#64748b",       # Slate Gray
        "badge_bg": "rgba(100, 116, 139, 0.15)",
        "instruction": "Place non-recyclable, multi-layered, or heavily soiled items into the general waste bin.",
        "eco_tip": "Reduce general waste by choosing reusable products and avoiding single-use mixed packaging.",
        "recyclable": False,
        "examples": ["Chip wrappers", "Styrofoam", "Soiled tissues", "Broken ceramics", "Mixed composite items"],
    },
}


def get_waste_rule(waste_type: str) -> Dict[str, Any]:
    """
    Retrieve the bin recommendation and disposal instructions for a given waste category.
    Falls back to 'General Waste' if an unrecognized category is passed.
    """
    normalized = waste_type.strip() if waste_type else "General Waste"
    # Case-insensitive lookup
    for key, rule in WASTE_RULES.items():
        if key.lower() == normalized.lower():
            return rule
    return WASTE_RULES["General Waste"]


def get_all_waste_rules() -> List[Dict[str, Any]]:
    """Return a list of all configured waste category rules."""
    return list(WASTE_RULES.values())
