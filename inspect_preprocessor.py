import joblib

preprocessor = joblib.load("models/preprocessor_v20251107_0700.pkl")

print("\n✅ Loaded preprocessor successfully.\n")
print("🔍 Class/type:", type(preprocessor))

# If it's a sklearn ColumnTransformer:
try:
    print("\n🧱 Transformers inside:")
    for name, transformer, cols in preprocessor.transformers_:
        print(f"• {name}: {type(transformer)} on columns: {cols}")
except:
    pass

# If it has feature names after transformation:
try:
    print("\n📋 Output feature names after transform:")
    print(preprocessor.get_feature_names_out())
except:
    print("\n⚠️ Preprocessor does not support get_feature_names_out()")
