import streamlit as st
from datetime import date

# --- Custom CSS for Dark‑Mode Cards & Layout ---
st.markdown("""
<style>
/* Base */
body, .main, .block-container {
  background-color: #0f172a;
  color: #e2e8f0;
}
h1, h2, h3, h4, h5, h6 {
  color: #f8fafc;
}

/* Sidebar */
[data-testid="stSidebar"] {
  background-color: #1e293b;
}
.stSidebar h2 {
  color: #38bdf8 !important;
  margin-bottom: 0.5rem;
}

/* Cards */
.card {
  background-color: #1e293b;
  border-radius: 8px;
  padding: 1rem;
  margin-bottom: 1rem;
  box-shadow: 0 2px 6px rgba(0,0,0,0.3);
}
.card-title {
  font-size: 1.1rem;
  color: #7dd3fc;
  margin-bottom: 0.25rem;
}
.card-text {
  color: #cbd5e1;
  margin: 0.15rem 0;
}
</style>
""", unsafe_allow_html=True)

# --- App Header ---
st.title("✈️ AI‑Powered Travel Planner")
st.write("A sleek, modular interface for flights, hotels, visas, videos & budgets.")

# --- Sidebar: To‑Do List ---
with st.sidebar:
    st.header("📝 To‑Do List")
    action = st.selectbox("Action", ["Create", "Read", "Update", "Delete"])
    item = st.text_input("To‑do item")
    if st.button("Execute"):
        st.info(f"{action} → {item!r}")

    st.markdown("---")
    st.subheader("Current Items")
    if "todos" not in st.session_state:
        st.session_state.todos = []
    for i, t in enumerate(st.session_state.todos):
        cols = st.columns([0.8, 0.1, 0.1])
        cols[0].write(f"- {t['task']}")
        if cols[1].button("✔️", key=f"done_{i}"):
            st.session_state.todos[i]["done"] = True
        if cols[2].button("❌", key=f"del_{i}"):
            st.session_state.todos.pop(i)
    st.write(f"📝 Total: {len(st.session_state.todos)}")    

# --- Step 1: Trip Inputs ---
st.header("1. Trip Details")
col1, col2 = st.columns(2)
with col1:
    city = st.text_input("Which city?", placeholder="e.g. Tokyo")
    trip_type = st.selectbox("Trip type", ["Round‑trip", "One‑way"])
with col2:
    dates = st.date_input("Dates", value=[date.today(), date.today()])
    if trip_type == "One‑way":
        # only first date matters
        dates = dates[:1]

search = st.button("🔍 Search Options")

# --- Placeholder Functions ---
def fetch_flight_options(city, d1, d2, ttype):
    return [
        {"airline":"AirSample","price":200,"depart":"08:00","arrive":"12:00"},
        {"airline":"DemoJet", "price":250,"depart":"14:00","arrive":"18:00"},
    ]
def fetch_hotel_options(city, d1, d2):
    return [
        {"name":"Hotel Alpha","rating":4.3,"rate":130},
        {"name":"Hotel Beta", "rating":4.7,"rate":160},
    ]
def fetch_visa(origin, dest):
    return {"type":"Tourist","duration":"90 days","cost":"$60"}
def fetch_videos(city):
    return [
        {"title":f"{city} Sights","thumb":"https://via.placeholder.com/150","url":"https://youtu.be/vid1"},
        {"title":f"{city} Tips","thumb":"https://via.placeholder.com/150","url":"https://youtu.be/vid2"},
    ]

# --- Step 2: Results ---
if search and city and dates:
    d1 = dates[0].strftime("%Y-%m-%d")
    d2 = dates[1].strftime("%Y-%m-%d") if trip_type=="Round‑trip" else None

    # Flights & Hotels Side‑by‑Side
    colF, colH = st.columns(2)
    with colF:
        st.subheader("🎫 Flight Options")
        flights = fetch_flight_options(city, d1, d2, trip_type)
        for f in flights:
            st.markdown("<div class='card'>", unsafe_allow_html=True)
            st.markdown(f"<div class='card-title'>{f['airline']} — ${f['price']}</div>", unsafe_allow_html=True)
            st.markdown(f"<div class='card-text'>🛫 {f['depart']} → 🛬 {f['arrive']}</div>", unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)

    with colH:
        st.subheader("🏨 Hotel Options")
        hotels = fetch_hotel_options(city, d1, d2)
        for h in hotels:
            st.markdown("<div class='card'>", unsafe_allow_html=True)
            st.markdown(f"<div class='card-title'>{h['name']} — ${h['rate']}/night</div>", unsafe_allow_html=True)
            st.markdown(f"<div class='card-text'>⭐ {h['rating']}</div>", unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)

    st.subheader("🛂 Visa Requirements")
    visa = fetch_visa("YourOrigin", city)
    st.info(f"{visa['type']} visa — up to {visa['duration']} — {visa['cost']}")

    st.subheader("📺 Top 5 Travel Videos")
    vids = fetch_videos(city)
    cols = st.columns(len(vids))
    for i, v in enumerate(vids):
        with cols[i]:
            st.image(v["thumb"], use_column_width=True)
            st.markdown(f"[{v['title']}]({v['url']})")

    st.subheader("💰 Budget Estimate")
    num = st.number_input("Travelers", min_value=1, value=1)
    nights = (dates[1] - dates[0]).days if trip_type=="Round‑trip" else 1
    avg_rate = sum(h["rate"] for h in hotels)/len(hotels)
    total = avg_rate * nights * num
    st.metric(label="Hotel‑only Budget", value=f"${total:,.2f}")
