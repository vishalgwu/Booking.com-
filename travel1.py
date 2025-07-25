import os
import requests
import json
import time
import re
from datetime import datetime, timedelta
from dotenv import load_dotenv
from openai import OpenAI
from dateutil import parser
from langchain_openai import ChatOpenAI
from typing import Annotated, Literal, Sequence, TypedDict
from langchain_core.tools import tool
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from pydantic import BaseModel, Field
from langgraph.graph.message import add_messages

# Load environment variables
load_dotenv()

# Initialize all required APIs
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.7)

# API Keys and Headers
RAPIDAPI_KEY = os.getenv("RAPIDAPI_KEY")
RAPIDAPI_HOST_HOTEL = os.getenv("RAPIDAPI_HOST_HOTEL")
RAPIDAPI_HOST_FLIGHT = os.getenv("RAPIDAPI_HOST_FLIGHT", "booking-com15.p.rapidapi.com")

HEADERS_HOTEL = {
    "x-rapidapi-key": RAPIDAPI_KEY,
    "x-rapidapi-host": RAPIDAPI_HOST_HOTEL
}

HEADERS_FLIGHT = {
    "x-rapidapi-key": RAPIDAPI_KEY,
    "x-rapidapi-host": RAPIDAPI_HOST_FLIGHT
}

# ==================== MAIN LLM COORDINATOR ====================

class TravelPlan(BaseModel):
    origin_city: str = Field(description="Departure city")
    destination_city: str = Field(description="Destination city") 
    departure_date: str = Field(description="Departure date in YYYY-MM-DD format")
    return_date: str = Field(description="Return date in YYYY-MM-DD format or null for one-way", default=None)
    adults: int = Field(description="Number of adults", default=1)
    children: int = Field(description="Number of children", default=0)
    party_type: str = Field(description="Type of travelers: solo, family, or friends")
    budget_preference: str = Field(description="Budget level: basic, advance, or business")
    cabin_class: str = Field(description="Flight cabin class: ECONOMY, PREMIUM_ECONOMY, BUSINESS, FIRST", default="ECONOMY")
    hotel_budget_min: float = Field(description="Minimum hotel budget per night", default=0)
    hotel_budget_max: float = Field(description="Maximum hotel budget per night", default=0)

def main_llm_coordinator(user_input: str) -> TravelPlan:
    """
    Main LLM that parses user input and extracts all required travel information
    """
    prompt = f"""
    You are an expert travel planning coordinator. Parse the user's travel request and extract comprehensive travel information.
    
    User input: "{user_input}"
    
    Extract and fill in the following information. If something is missing, make reasonable assumptions or set defaults:
    
    Please respond with ONLY valid JSON in this exact format:
    {{
        "origin_city": "<departure city>",
        "destination_city": "<destination city>", 
        "departure_date": "<YYYY-MM-DD format>",
        "return_date": "<YYYY-MM-DD format or null for one-way>",
        "adults": <number of adults>,
        "children": <number of children>,
        "party_type": "<solo|family|friends>",
        "budget_preference": "<basic|advance|business>",
        "cabin_class": "<ECONOMY|PREMIUM_ECONOMY|BUSINESS|FIRST>",
        "hotel_budget_min": <minimum hotel budget per night in USD or 0 if not specified>,
        "hotel_budget_max": <maximum hotel budget per night in USD or 0 if not specified>
    }}
    
    Rules:
    - If dates are relative (like "next week", "tomorrow"), convert to actual YYYY-MM-DD format
    - If return date not specified, set to null for one-way trip
    - If party type unclear, infer from context (1 person = solo, family words = family, multiple friends = friends)
    - If budget not specified, default to "advance" 
    - If cabin class not specified, default to "ECONOMY"
    - If hotel budget not specified, set both min and max to 0
    - Current date for reference: {datetime.now().strftime('%Y-%m-%d')}
    
    Examples:
    - "I want to go from New York to Paris next month with my family, business class, budget around $300/night for hotel" 
    - "Solo trip to Tokyo from LA, leaving December 1st, returning Dec 10th, budget travel"
    - "Friends trip to Goa from Mumbai, 3 people, mid-range budget"
    """
    
    response = client.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3
    )
    
    try:
        data = json.loads(response.choices[0].message.content)
        print(f"✅ LLM successfully parsed travel requirements")
        return TravelPlan(**data)
    except (json.JSONDecodeError, Exception) as e:
        print(f"⚠️ LLM parsing failed, using interactive input: {e}")
        return get_interactive_plan(user_input)

def get_interactive_plan(user_input: str) -> TravelPlan:
    """Interactive input collection when LLM fails"""
    print("\n📝 Let's gather your travel details step by step...")
    
    # Try to extract basic info from user input
    words = user_input.lower().split()
    
    # Smart defaults based on user input
    default_origin = ""
    default_dest = ""
    default_date = ""
    default_adults = 1
    default_party = "solo"
    default_budget = "advance"
    
    # Extract cities if "from" and "to" are present
    if "from" in words:
        from_idx = words.index("from")
        if from_idx + 1 < len(words):
            # Take next 1-2 words as origin
            default_origin = " ".join(words[from_idx+1:from_idx+3]).replace("to", "").strip()
    
    if "to" in words:
        to_idx = words.index("to")
        if to_idx + 1 < len(words):
            # Take next 1-2 words as destination
            default_dest = " ".join(words[to_idx+1:to_idx+3]).strip()
    
    # Extract date patterns
    import re
    date_pattern = r'\d{4}-\d{2}-\d{2}'
    dates = re.findall(date_pattern, user_input)
    if dates:
        default_date = dates[0]
    
    # Extract number of people
    numbers = re.findall(r'\b(\d+)\s*(?:people|members|persons|adults|travelers)\b', user_input.lower())
    if numbers:
        default_adults = int(numbers[0])
        default_party = "family" if default_adults > 2 else "friends" if default_adults > 1 else "solo"
    
    # Extract budget level
    if any(word in user_input.lower() for word in ["basic", "budget", "cheap", "economy"]):
        default_budget = "basic"
    elif any(word in user_input.lower() for word in ["business", "luxury", "premium", "first"]):
        default_budget = "business"
    
    # Collect missing information
    origin = input(f"From which city? {f'[{default_origin}]: ' if default_origin else ''}")
    origin = origin.strip() or default_origin
    
    dest = input(f"To which city? {f'[{default_dest}]: ' if default_dest else ''}")
    dest = dest.strip() or default_dest
    
    dep_date = input(f"Departure date (YYYY-MM-DD)? {f'[{default_date}]: ' if default_date else ''}")
    dep_date = dep_date.strip() or default_date or (datetime.now() + timedelta(days=30)).strftime("%Y-%m-%d")
    
    ret_date = input("Return date (YYYY-MM-DD) or press Enter for one-way: ").strip() or None
    
    adults_input = input(f"Number of adults? [{default_adults}]: ")
    adults = int(adults_input) if adults_input.strip() else default_adults
    
    children_input = input("Number of children? [0]: ")
    children = int(children_input) if children_input.strip() else 0
    
    party_input = input(f"Travel type (solo/family/friends)? [{default_party}]: ")
    party = party_input.strip() or default_party
    
    budget_input = input(f"Budget level (basic/advance/business)? [{default_budget}]: ")
    budget = budget_input.strip() or default_budget
    
    cabin_input = input("Flight class (ECONOMY/BUSINESS/FIRST)? [ECONOMY]: ")
    cabin = cabin_input.strip() or "ECONOMY"
    
    return TravelPlan(
        origin_city=origin,
        destination_city=dest,
        departure_date=dep_date,
        return_date=ret_date,
        adults=adults,
        children=children,
        party_type=party,
        budget_preference=budget,
        cabin_class=cabin,
        hotel_budget_min=0,
        hotel_budget_max=0
    )

# ==================== FLIGHT SEARCH MODULE ====================

def llm_get_airport_code(city_name: str) -> dict:
    """Use LLM to get airport code for a city"""
    prompt = f"""Find the best airport for: '{city_name}'
    
    Respond with ONLY valid JSON:
    {{
        "code": "<3-letter IATA code>",
        "airport_name": "<Full airport name>",
        "city": "<City>",
        "country": "<Country>"
    }}"""
    
    response = client.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2
    )
    
    try:
        return json.loads(response.choices[0].message.content)
    except:
        return {"code": "UNK", "airport_name": "Unknown", "city": city_name, "country": "Unknown"}

def search_flights(origin_code: str, dest_code: str, depart_date: str, cabin: str = "ECONOMY", adults: int = 1, children: int = 0) -> list[dict]:
    """Search for flights using Booking.com API"""
    try:
        if not origin_code.endswith('.AIRPORT'):
            origin_code = f"{origin_code}.AIRPORT"
        if not dest_code.endswith('.AIRPORT'):
            dest_code = f"{dest_code}.AIRPORT"
        
        url = f"https://{RAPIDAPI_HOST_FLIGHT}/api/v1/flights/searchFlights"
        
        params = {
            "fromId": origin_code,
            "toId": dest_code,
            "departDate": depart_date,
            "pageNo": 1,
            "adults": adults,
            "children": "0%2C17" if children > 0 else "0%2C17",
            "sort": "BEST",
            "cabinClass": cabin.upper(),
            "currency_code": "USD"
        }
        
        print(f"🔍 Searching flights from {origin_code} to {dest_code}...")
        time.sleep(2)  # Rate limiting
        
        resp = requests.get(url, headers=HEADERS_FLIGHT, params=params, timeout=30)
        
        if resp.status_code == 429:
            print("⏳ Rate limit, waiting...")
            time.sleep(10)
            resp = requests.get(url, headers=HEADERS_FLIGHT, params=params, timeout=30)
        
        resp.raise_for_status()
        response_data = resp.json()
        
        return parse_flight_response(response_data)
        
    except Exception as e:
        print(f"❌ Flight search error: {e}")
        return []

def parse_flight_response(data: dict) -> list[dict]:
    """Parse flight API response"""
    flights = []
    
    if not data.get("status"):
        return flights
        
    flight_data = data.get("data", {})
    offers = flight_data.get("flightOffers", [])
    
    for offer in offers:
        try:
            segments = offer.get("segments", [])
            if not segments:
                continue
                
            first_segment = segments[0]
            legs = first_segment.get("legs", [])
            if not legs:
                continue
                
            leg = legs[0]
            carriers = leg.get("carriersData", [])
            carrier_name = carriers[0].get("name", "Unknown") if carriers else "Unknown"
            
            flight_info = {
                "departure_time": leg.get("departureTime", ""),
                "arrival_time": leg.get("arrivalTime", ""),
                "from_code": first_segment.get("departureAirport", {}).get("code", ""),
                "to_code": first_segment.get("arrivalAirport", {}).get("code", ""),
                "carrier": carrier_name,
                "flight_number": leg.get("flightInfo", {}).get("flightNumber", ""),
                "duration_minutes": first_segment.get("totalTime", 0) // 60,
                "is_direct": len(segments) == 1 and len(legs) == 1,
                "stops": max(0, len(segments) - 1),
                "cabin_class": leg.get("cabinClass", "ECONOMY"),
                "price": offer.get("priceBreakdown", {}).get("total", {}).get("value", "N/A")
            }
            flights.append(flight_info)
            
        except Exception as e:
            continue
    
    return flights

# ==================== IMPROVED HOTEL SEARCH MODULE ====================

def search_destination(query: str, auto_select: bool = True) -> str:
    """
    Enhanced destination search with better error handling and multiple options
    """
    try:
        url = f"https://{RAPIDAPI_HOST_HOTEL}/api/v1/hotels/searchDestination"
        params = {"query": query}
        
        print(f"🔍 Searching destinations for: '{query}'")
        
        resp = requests.get(url, headers=HEADERS_HOTEL, params=params, timeout=15)
        resp.raise_for_status()
        data = resp.json()
        
        print(f"🔍 API Response Status: {data.get('status')}")
        
        if not data.get("status"):
            error_msg = data.get('message', 'Unknown API error')
            print(f"❌ API returned error: {error_msg}")
            
            # Try alternative search terms
            alternative_queries = generate_alternative_queries(query)
            for alt_query in alternative_queries:
                print(f"🔄 Trying alternative search: '{alt_query}'")
                try:
                    alt_params = {"query": alt_query}
                    alt_resp = requests.get(url, headers=HEADERS_HOTEL, params=alt_params, timeout=15)
                    alt_data = alt_resp.json()
                    if alt_data.get("status") and alt_data.get("data"):
                        data = alt_data
                        break
                except:
                    continue
        
        results = data.get("data", [])
        if not results:
            raise ValueError(f"No destinations found for '{query}' or alternatives")
        
        print(f"✅ Found {len(results)} destinations")
        
        # Display options for better selection
        if not auto_select and len(results) > 1:
            print("\n🌍 Available destinations:")
            for i, dest in enumerate(results[:5], 1):  # Show top 5
                country = dest.get('country', 'Unknown Country')
                region = dest.get('region', '')
                hotels_count = dest.get('hotels', 0)
                
                print(f"{i}. {dest['name']}")
                print(f"   🌍 Country: {country}")
                if region:
                    print(f"   📍 Region: {region}")
                print(f"   🏨 Available hotels: {hotels_count}")
                print(f"   🆔 ID: {dest['dest_id']}")
                print("-" * 50)
            
            try:
                choice = int(input(f"Select destination (1-{min(5, len(results))}): ")) - 1
                if 0 <= choice < len(results):
                    selected = results[choice]
                else:
                    selected = results[0]
            except:
                selected = results[0]
        else:
            selected = results[0]
        
        print(f"✅ Selected: {selected['name']} (ID: {selected['dest_id']})")
        return selected["dest_id"]
        
    except Exception as e:
        print(f"❌ Destination search failed: {e}")
        raise ValueError(f"Could not find destination for '{query}': {e}")

def generate_alternative_queries(original_query: str) -> list[str]:
    """Generate alternative search queries for better destination matching"""
    alternatives = []
    
    # Add common variations
    query_lower = original_query.lower()
    
    # Add country suffixes
    common_countries = ["france", "usa", "uk", "india", "italy", "spain", "germany", "japan"]
    for country in common_countries:
        if country not in query_lower:
            alternatives.append(f"{original_query} {country}")
    
    # Add city variations
    city_variations = {
        "paris": ["paris france"],
        "london": ["london uk", "london england"],
        "new york": ["new york city", "nyc", "new york usa"],
        "washington": ["washington dc", "washington d.c."],
        "mumbai": ["mumbai india", "bombay"],
        "delhi": ["new delhi", "delhi india"]
    }
    
    for city, variations in city_variations.items():
        if city in query_lower:
            alternatives.extend(variations)
    
    # Remove duplicates and original query
    alternatives = list(set(alternatives))
    if original_query in alternatives:
        alternatives.remove(original_query)
    
    return alternatives[:3]  # Return top 3 alternatives

def validate_date_format(date_str: str) -> str:
    """Validate and format date string"""
    try:
        # Try parsing the date
        parsed_date = datetime.strptime(date_str, "%Y-%m-%d")
        return parsed_date.strftime("%Y-%m-%d")
    except ValueError:
        try:
            # Try alternative formats
            parsed_date = parser.parse(date_str)
            return parsed_date.strftime("%Y-%m-%d")
        except:
            # Return today's date as fallback
            return datetime.now().strftime("%Y-%m-%d")

def list_hotels(dest_id: str, checkin: str, checkout: str, adults: int = 1, limit: int = 30) -> list[dict]:
    """
    Enhanced hotel search with better error handling and debugging
    """
    try:
        # Validate and format dates
        checkin = validate_date_format(checkin)
        checkout = validate_date_format(checkout)
        
        # Ensure checkout is after checkin
        checkin_date = datetime.strptime(checkin, "%Y-%m-%d")
        checkout_date = datetime.strptime(checkout, "%Y-%m-%d")
        
        if checkout_date <= checkin_date:
            checkout_date = checkin_date + timedelta(days=1)
            checkout = checkout_date.strftime("%Y-%m-%d")
            print(f"⚠️ Adjusted checkout date to: {checkout}")
        
        url = f"https://{RAPIDAPI_HOST_HOTEL}/api/v1/hotels/searchHotels"
        
        params = {
            "dest_id": str(dest_id),
            "search_type": "city",
            "arrival_date": checkin,
            "departure_date": checkout,
            "adults_number": str(adults),
            "room_number": "1",
            "locale": "en-us",
            "currency": "USD",
            "order_by": "popularity"
        }
        
        print(f"🏨 Searching hotels with parameters:")
        print(f"   📍 Destination ID: {dest_id}")
        print(f"   📅 Check-in: {checkin}")
        print(f"   📅 Check-out: {checkout}")
        print(f"   👥 Adults: {adults}")
        
        # Add delays to avoid rate limiting
        time.sleep(1)
        
        resp = requests.get(url, headers=HEADERS_HOTEL, params=params, timeout=20)
        
        print(f"   📊 HTTP Status: {resp.status_code}")
        
        if resp.status_code == 429:
            print("⏳ Rate limit hit, waiting 30 seconds...")
            time.sleep(30)
            resp = requests.get(url, headers=HEADERS_HOTEL, params=params, timeout=20)
        
        if resp.status_code == 200:
            try:
                data = resp.json()
                print(f"   🔍 API Response Status: {data.get('status')}")
                
                if data.get("status"):
                    data_obj = data.get("data", {})
                    hotel_results = data_obj.get("hotels", [])
                    
                    print(f"   📊 Raw hotel count: {len(hotel_results)}")
                    
                    if hotel_results:
                        parsed_hotels = parse_hotels_enhanced(hotel_results, limit)
                        print(f"   ✅ Successfully parsed {len(parsed_hotels)} hotels")
                        return parsed_hotels
                    else:
                        print(f"   ❌ No hotels found in API response")
                        # Try alternative search parameters
                        return try_alternative_hotel_search(dest_id, checkin, checkout, adults)
                else:
                    error_msg = data.get('message', 'Unknown API error')
                    print(f"   ❌ API error: {error_msg}")
                    return []
                    
            except json.JSONDecodeError as e:
                print(f"   ❌ JSON parsing error: {e}")
                print(f"   📄 Response preview: {resp.text[:300]}...")
                return []
                
        else:
            print(f"   ❌ HTTP Error {resp.status_code}")
            print(f"   📄 Response preview: {resp.text[:200]}...")
            return []
            
    except requests.exceptions.RequestException as e:
        print(f"   ❌ Request failed: {e}")
        return []
    except Exception as e:
        print(f"   ❌ Unexpected error: {e}")
        return []

def try_alternative_hotel_search(dest_id: str, checkin: str, checkout: str, adults: int) -> list[dict]:
    """Try alternative search parameters when initial search fails"""
    print("🔄 Trying alternative hotel search parameters...")
    
    alternatives = [
        # Try different search types
        {"search_type": "region", "order_by": "price"},
        {"search_type": "landmark", "order_by": "distance"},
        # Try adjusting dates slightly
        {"arrival_date": (datetime.strptime(checkin, "%Y-%m-%d") + timedelta(days=1)).strftime("%Y-%m-%d"),
         "departure_date": (datetime.strptime(checkout, "%Y-%m-%d") + timedelta(days=1)).strftime("%Y-%m-%d")},
        # Try without specific locale
        {"locale": "en-gb"},
    ]
    
    base_params = {
        "dest_id": str(dest_id),
        "search_type": "city",
        "arrival_date": checkin,
        "departure_date": checkout,
        "adults_number": str(adults),
        "room_number": "1",
        "currency": "USD",
        "order_by": "popularity"
    }
    
    for i, alt_params in enumerate(alternatives, 1):
        try:
            print(f"   🔄 Alternative {i}: {alt_params}")
            
            # Merge with base parameters
            params = {**base_params, **alt_params}
            
            url = f"https://{RAPIDAPI_HOST_HOTEL}/api/v1/hotels/searchHotels"
            
            time.sleep(2)  # Rate limiting
            resp = requests.get(url, headers=HEADERS_HOTEL, params=params, timeout=15)
            
            if resp.status_code == 200:
                data = resp.json()
                if data.get("status"):
                    hotel_results = data.get("data", {}).get("hotels", [])
                    if hotel_results:
                        print(f"   ✅ Alternative {i} found {len(hotel_results)} hotels!")
                        return parse_hotels_enhanced(hotel_results, 20)
            
        except Exception as e:
            print(f"   ❌ Alternative {i} failed: {e}")
            continue
    
    print("   ❌ All alternatives exhausted")
    return []

def parse_hotels_enhanced(hotel_results: list, limit: int) -> list[dict]:
    """Enhanced hotel parsing with better field extraction"""
    hotels = []
    
    print(f"🔍 Parsing {len(hotel_results)} hotel results...")
    
    for i, hotel_data in enumerate(hotel_results[:limit]):
        try:
            # Method 1: Try accessibilityLabel parsing (most reliable)
            parsed_hotel = parse_from_accessibility_label(hotel_data)
            
            if not parsed_hotel:
                # Method 2: Try property object parsing
                parsed_hotel = parse_from_property_object(hotel_data)
            
            if not parsed_hotel:
                # Method 3: Try direct field parsing
                parsed_hotel = parse_from_direct_fields(hotel_data)
            
            if parsed_hotel:
                hotels.append(parsed_hotel)
            else:
                print(f"   ⚠️ Could not parse hotel {i+1}")
                
        except Exception as e:
            print(f"   ❌ Error parsing hotel {i+1}: {e}")
            continue
    
    print(f"   ✅ Successfully parsed {len(hotels)} hotels")
    return hotels

def parse_from_accessibility_label(hotel_data: dict) -> dict:
    """Parse hotel info from accessibilityLabel (primary method)"""
    try:
        accessibility_label = hotel_data.get('accessibilityLabel', '')
        if not accessibility_label:
            return None
        
        lines = accessibility_label.split('\n')
        if not lines:
            return None
        
        # Extract hotel name (usually first line)
        name = lines[0].strip() if lines else "Unknown Hotel"
        
        # Extract address/location (look for 'km from centre' or location info)
        address = "Address not available"
        for line in lines[1:]:
            if 'km from centre' in line or 'from center' in line:
                address = line.strip()
                break
            elif any(keyword in line.lower() for keyword in ['street', 'avenue', 'road', 'district', 'area']):
                address = line.strip()
                break
        
        # Extract price (look for USD or currency symbols)
        price = "Price not available"
        price_patterns = [
            r'(\$?\d{1,4})\s*USD',
            r'USD\s*(\$?\d{1,4})',
            r'\$(\d{1,4})',
            r'(\d{1,4})\s*dollars?'
        ]
        
        for pattern in price_patterns:
            match = re.search(pattern, accessibility_label, re.IGNORECASE)
            if match:
                price_value = match.group(1).replace('$', '')
                price = f"${price_value} USD"
                break
        
        # Extract rating if available
        rating = "Not rated"
        rating_match = re.search(r'(\d+(?:\.\d+)?)\s*(?:star|stars|out of)', accessibility_label, re.IGNORECASE)
        if rating_match:
            rating = f"{rating_match.group(1)} stars"
        
        return {
            "name": name,
            "address": address,
            "price": price,
            "rating": rating,
            "hotel_id": hotel_data.get("hotel_id", "N/A"),
            "source": "accessibility_label"
        }
        
    except Exception as e:
        return None

def parse_from_property_object(hotel_data: dict) -> dict:
    """Parse hotel info from property object (secondary method)"""
    try:
        property_obj = hotel_data.get('property', {})
        if not property_obj:
            return None
        
        name = property_obj.get('name', 'Unknown Hotel')
        
        # Try to get address from various fields
        address_parts = []
        if property_obj.get('address'):
            address_parts.append(property_obj['address'])
        if property_obj.get('city'):
            address_parts.append(property_obj['city'])
        if property_obj.get('country'):
            address_parts.append(property_obj['country'])
        
        address = ', '.join(address_parts) if address_parts else "Address not available"
        
        # Try to get price
        price = "Price not available"
        if 'price' in property_obj:
            price_obj = property_obj['price']
            if isinstance(price_obj, dict):
                if 'amount' in price_obj:
                    currency = price_obj.get('currency', 'USD')
                    price = f"${price_obj['amount']} {currency}"
            elif isinstance(price_obj, (int, float)):
                price = f"${price_obj} USD"
        
        # Try to get rating
        rating = "Not rated"
        if 'rating' in property_obj:
            rating = f"{property_obj['rating']} stars"
        
        return {
            "name": name,
            "address": address,
            "price": price,
            "rating": rating,
            "hotel_id": hotel_data.get("hotel_id", property_obj.get("id", "N/A")),
            "source": "property_object"
        }
        
    except Exception as e:
        return None

def parse_from_direct_fields(hotel_data: dict) -> dict:
    """Parse hotel info from direct fields (fallback method)"""
    try:
        # Try common field names
        name_fields = ['name', 'hotel_name', 'title', 'property_name']
        name = "Unknown Hotel"
        for field in name_fields:
            if field in hotel_data and hotel_data[field]:
                name = str(hotel_data[field]).strip()
                break
        
        # Try address fields
        address_fields = ['address', 'location', 'area', 'district']
        address = "Address not available"
        for field in address_fields:
            if field in hotel_data and hotel_data[field]:
                address = str(hotel_data[field]).strip()
                break
        
        # Try price fields
        price_fields = ['price', 'rate', 'cost', 'amount']
        price = "Price not available"
        for field in price_fields:
            if field in hotel_data and hotel_data[field]:
                price_val = hotel_data[field]
                if isinstance(price_val, (int, float)):
                    price = f"${price_val} USD"
                elif isinstance(price_val, str) and any(c.isdigit() for c in price_val):
                    price = price_val
                break
        
        return {
            "name": name,
            "address": address,
            "price": price,
            "rating": "Not rated",
            "hotel_id": hotel_data.get("hotel_id", hotel_data.get("id", "N/A")),
            "source": "direct_fields"
        }
        
    except Exception as e:
        return None

def extract_price_value(price_str: str) -> float:
    """
    Extract numeric price value from price string.
    Examples: "$1648 USD" -> 1648.0, "Price not available" -> None
    """
    if not price_str or price_str == "Price not available":
        return None
    
    # Use regex to find numbers in the price string
    price_match = re.search(r'(\d+(?:\.\d+)?)', price_str.replace(',', ''))
    if price_match:
        return float(price_match.group(1))
    return None

def filter_hotels_by_budget(hotels: list[dict], min_budget: float = 0, max_budget: float = 0, nights: int = 1) -> list[dict]:
    """
    Enhanced budget filtering with better price extraction and per-night calculation
    """
    if min_budget == 0 and max_budget == 0:
        return hotels
    
    filtered_hotels = []
    skipped_count = 0
    
    print(f"\n💰 Filtering hotels by budget (per night):")
    print(f"   📅 Stay duration: {nights} nights")
    if min_budget > 0:
        print(f"   💵 Minimum per night: ${min_budget} (Total: ${min_budget * nights})")
    if max_budget > 0:
        print(f"   💵 Maximum per night: ${max_budget} (Total: ${max_budget * nights})")
    
    for hotel in hotels:
        total_price = extract_price_value(hotel['price'])
        
        if total_price is None:
            print(f"   ⚠️ Skipping {hotel['name'][:30]}... (no price available)")
            skipped_count += 1
            # Include hotels without price if no strict budget constraints
            if min_budget == 0 and max_budget == 0:
                filtered_hotels.append(hotel)
            continue
        
        # Calculate per-night price
        per_night_price = total_price / max(nights, 1)
        
        # Check budget constraints (per night)
        within_budget = True
        
        if min_budget > 0 and per_night_price < min_budget:
            within_budget = False
        
        if max_budget > 0 and per_night_price > max_budget:
            within_budget = False
        
        if within_budget:
            # Add per-night info to hotel for display
            hotel_copy = hotel.copy()
            hotel_copy['per_night_price'] = f"${per_night_price:.0f}/night"
            filtered_hotels.append(hotel_copy)
            print(f"   ✅ {hotel['name'][:40]}... - ${per_night_price:.0f}/night (${total_price:.0f} total)")
        else:
            print(f"   ❌ {hotel['name'][:40]}... - ${per_night_price:.0f}/night (${total_price:.0f} total) - outside budget")
    
    print(f"\n📊 Budget filtering results:")
    print(f"   ✅ {len(filtered_hotels)} hotels within budget")
    print(f"   ❌ {len(hotels) - len(filtered_hotels) - skipped_count} hotels outside budget")
    if skipped_count > 0:
        print(f"   ⚠️ {skipped_count} hotels skipped (no price)")
    
    return filtered_hotels

def categorize_hotels_by_budget(hotels: list[dict], checkin: str = None, checkout: str = None) -> dict:
    """
    Categorize hotels into budget ranges based on per-night pricing.
    """
    # Calculate number of nights
    nights = 1
    if checkin and checkout:
        try:
            checkin_date = datetime.strptime(checkin, '%Y-%m-%d')
            checkout_date = datetime.strptime(checkout, '%Y-%m-%d')
            nights = max((checkout_date - checkin_date).days, 1)
        except ValueError:
            nights = 1
    
    categories = {
        "budget": {"range": "Under $100/night", "hotels": []},
        "mid_range": {"range": "$100 - $200/night", "hotels": []},
        "upscale": {"range": "$200 - $400/night", "hotels": []},
        "luxury": {"range": "$400+/night", "hotels": []}
    }
    
    for hotel in hotels:
        total_price = extract_price_value(hotel['price'])
        
        if total_price is None:
            continue
        
        per_night_price = total_price / nights
        
        # Add per-night price info to hotel
        hotel_with_per_night = hotel.copy()
        hotel_with_per_night['per_night_price'] = f"${per_night_price:.0f}/night"
        hotel_with_per_night['total_price'] = hotel['price']
        
        if per_night_price < 100:
            categories["budget"]["hotels"].append(hotel_with_per_night)
        elif per_night_price < 200:
            categories["mid_range"]["hotels"].append(hotel_with_per_night)
        elif per_night_price < 400:
            categories["upscale"]["hotels"].append(hotel_with_per_night)
        else:
            categories["luxury"]["hotels"].append(hotel_with_per_night)
    
    return categories

# ==================== ITINERARY AGENT MODULE ====================

class TripState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    origin: str
    destination: str
    start_date: str
    end_date: str
    party_type: str
    num_people: int
    budget_preference: str
    trip_info: dict

def calculate_trip_duration(start_date: str, end_date: str) -> int:
    """Calculate trip duration"""
    try:
        start = parser.parse(start_date).date()
        end = parser.parse(end_date).date()
        return (end - start).days + 1
    except:
        return 1

@tool
def generate_packing_list(origin: str, destination: str, start_date: str, end_date: str, party_type: str, num_people: int) -> dict:
    """Generate packing list using LLM"""
    prompt = PromptTemplate.from_template("""
    Generate a comprehensive packing list for:
    Origin: {origin}
    Destination: {destination}
    Start Date: {start_date}
    End Date: {end_date}
    Party Type: {party_type}
    Number of People: {num_people}
    
    Consider weather, activities, and duration. List 10-15 essential items.
    """)
    
    formatted_prompt = prompt.format(
        origin=origin, destination=destination, start_date=start_date,
        end_date=end_date, party_type=party_type, num_people=num_people
    )
    response = llm.invoke(formatted_prompt)
    
    # Parse items
    lines = response.content.strip().split('\n')
    items = []
    for line in lines:
        line = line.strip()
        if line and (line[0].isdigit() or line.startswith('-') or line.startswith('•')):
            item = line.split('.', 1)[-1].strip() if '.' in line else line.strip('-• ')
            if item:
                items.append(item)
    
    return {"items": items}

@tool
def generate_itinerary(origin: str, destination: str, start_date: str, end_date: str, party_type: str, num_people: int) -> dict:
    """Generate day-by-day itinerary"""
    prompt = PromptTemplate.from_template("""
    Create a detailed day-by-day itinerary for:
    Origin: {origin}
    Destination: {destination}
    Start Date: {start_date}
    End Date: {end_date}
    Party Type: {party_type}
    Number of People: {num_people}
    
    Include attractions, activities, restaurants, and logistics.
    Consider museums, monuments, local experiences, food, sports, adventure activities.
    """)
    
    response = llm.invoke(prompt.format(
        origin=origin, destination=destination, start_date=start_date,
        end_date=end_date, party_type=party_type, num_people=num_people
    ))
    
    return {"itinerary": response.content}

@tool
def budget_calculator(origin: str, destination: str, start_date: str, end_date: str, 
                     party_type: str, num_people: int, budget_preference: str) -> dict:
    """Calculate detailed budget breakdown"""
    duration = calculate_trip_duration(start_date, end_date)
    nights = max(duration - 1, 0)
    
    prompt = PromptTemplate.from_template("""
    Calculate detailed budget for:
    Origin: {origin}
    Destination: {destination}
    Duration: {duration} days
    Party Type: {party_type}
    People: {num_people}
    Budget Level: {budget_preference}
    
    Budget Levels:
    - BASIC: Budget options, economy flights, 3-star hotels
    - ADVANCE: Mid-range, premium economy, 4-star hotels  
    - BUSINESS: Premium, business class, 5-star hotels
    
    Calculate: Flights, Accommodation ({nights} nights), Food, Activities, Transport, Miscellaneous
    Format as detailed breakdown with totals.
    """)
    
    response = llm.invoke(prompt.format(
        origin=origin, destination=destination, duration=duration,
        nights=nights, party_type=party_type, num_people=num_people,
        budget_preference=budget_preference
    ))
    
    return {"budget_breakdown": response.content, "duration": duration}

# ==================== MAIN ORCHESTRATOR ====================

def run_complete_travel_planner():
    """Main function that orchestrates all travel planning modules"""
    print("🌍 AI-Powered Complete Travel Planning System")
    print("=" * 60)
    
    # Get user input
    user_input = input("🎯 Describe your travel plans: ").strip()
    
    if not user_input:
        print("❌ Please provide travel details")
        return
    
    print("\n🤖 Processing your request...")
    
    # Step 1: Parse user input with main LLM
    travel_plan = main_llm_coordinator(user_input)
    
    print(f"\n📋 Parsed Travel Plan:")
    print(f"   From: {travel_plan.origin_city}")
    print(f"   To: {travel_plan.destination_city}")
    print(f"   Departure: {travel_plan.departure_date}")
    print(f"   Return: {travel_plan.return_date or 'One-way'}")
    print(f"   Travelers: {travel_plan.adults} adults, {travel_plan.children} children")
    print(f"   Party Type: {travel_plan.party_type}")
    print(f"   Budget Level: {travel_plan.budget_preference}")
    print(f"   Flight Class: {travel_plan.cabin_class}")
    
    # Step 2: Flight Search
    print(f"\n✈️ SEARCHING FLIGHTS...")
    print("-" * 40)
    
    origin_airport = llm_get_airport_code(travel_plan.origin_city)
    dest_airport = llm_get_airport_code(travel_plan.destination_city)
    
    print(f"🛫 {origin_airport['code']} - {origin_airport['airport_name']}")
    print(f"🛬 {dest_airport['code']} - {dest_airport['airport_name']}")
    
    flights = search_flights(
        origin_airport['code'],
        dest_airport['code'],
        travel_plan.departure_date,
        travel_plan.cabin_class,
        travel_plan.adults,
        travel_plan.children
    )
    
    # Display flights
    if flights:
        print(f"\n✅ Found {len(flights)} flights:")
        print(f"{'Time':<12}{'Route':<12}{'Carrier':<20}{'Duration':<10}{'Stops':<10}{'Price':<10}")
        print("-" * 80)
        
        for flight in flights[:5]:  # Show top 5
            dep_time = flight["departure_time"][11:16] if len(flight["departure_time"]) > 10 else flight["departure_time"][:5]
            route = f"{flight['from_code']}→{flight['to_code']}"
            duration = f"{flight['duration_minutes']}m" if flight['duration_minutes'] > 0 else "N/A"
            stops = "Direct" if flight["is_direct"] else f"{flight['stops']} stop"
            price = f"${flight['price']}" if flight['price'] != "N/A" else "N/A"
            
            print(f"{dep_time:<12}{route:<12}{flight['carrier'][:19]:<20}{duration:<10}{stops:<10}{price:<10}")
    else:
        print("❌ No flights found")
    
    # Step 3: Enhanced Hotel Search
    print(f"\n🏨 SEARCHING HOTELS...")
    print("-" * 40)
    
    try:
        # Use enhanced destination search
        dest_id = search_destination(travel_plan.destination_city, auto_select=True)
        
        # Use return date if available, otherwise use departure date + 1 day
        checkout_date = travel_plan.return_date
        if not checkout_date:
            checkout_date = (datetime.strptime(travel_plan.departure_date, "%Y-%m-%d") + timedelta(days=1)).strftime("%Y-%m-%d")
        
        # Calculate nights for budget filtering
        nights = calculate_trip_duration(travel_plan.departure_date, checkout_date) - 1
        nights = max(nights, 1)  # At least 1 night
        
        # Search hotels with enhanced method
        hotels = list_hotels(
            dest_id,
            travel_plan.departure_date,
            checkout_date,
            travel_plan.adults,
            limit=30  # Increased limit for better results
        )
        
        print(f"🔍 Initial hotel search returned {len(hotels)} hotels")
        
        # Apply budget filter if specified
        if travel_plan.hotel_budget_min > 0 or travel_plan.hotel_budget_max > 0:
            original_count = len(hotels)
            hotels = filter_hotels_by_budget(
                hotels, 
                travel_plan.hotel_budget_min, 
                travel_plan.hotel_budget_max, 
                nights
            )
            print(f"💰 After budget filtering: {len(hotels)} hotels (was {original_count})")
        
        # Display hotels
        if hotels:
            print(f"\n✅ Top {min(10, len(hotels))} hotels in {travel_plan.destination_city}:")
            print("=" * 80)
            
            for i, hotel in enumerate(hotels[:10], 1):
                print(f"{i}. {hotel['name']}")
                print(f"    📍 {hotel['address']}")
                print(f"    💰 {hotel['price']}")
                if 'per_night_price' in hotel:
                    print(f"    🌙 {hotel['per_night_price']}")
                if hotel.get('rating', 'Not rated') != 'Not rated':
                    print(f"    ⭐ {hotel['rating']}")
                if hotel['hotel_id'] != "N/A":
                    print(f"    🏨 ID: {hotel['hotel_id']}")
                print(f"    📊 Source: {hotel.get('source', 'unknown')}")
                print("-" * 60)
            
            # Show budget categories if many hotels found
            if len(hotels) > 5:
                print(f"\n🏷️ HOTELS BY BUDGET CATEGORY:")
                print("-" * 40)
                categories = categorize_hotels_by_budget(hotels, travel_plan.departure_date, checkout_date)
                
                for category, data in categories.items():
                    if data["hotels"]:
                        print(f"\n💰 {data['range'].upper()} ({len(data['hotels'])} hotels)")
                        for hotel in data["hotels"][:3]:  # Show top 3 in each category
                            print(f"   • {hotel['name']} - {hotel.get('per_night_price', hotel['price'])}")
        else:
            print("❌ No hotels found matching criteria")
            print("💡 Suggestions:")
            print("   • Try adjusting your travel dates")
            print("   • Consider a broader destination search")
            print("   • Remove or adjust budget constraints")
            print("   • Check if the destination name is correct")
            
    except Exception as e:
        print(f"❌ Hotel search error: {e}")
        print("💡 This might be due to:")
        print("   • API rate limits or temporary issues")
        print("   • Destination not found in hotel database")
        print("   • Network connectivity issues")
    
    # Step 4: Generate Itinerary and Budget
    print(f"\n📝 GENERATING TRAVEL PLAN...")
    print("-" * 40)
    
    # Create state for itinerary agent
    trip_state = {
        "messages": [],
        "origin": travel_plan.origin_city,
        "destination": travel_plan.destination_city,
        "start_date": travel_plan.departure_date,
        "end_date": travel_plan.return_date or travel_plan.departure_date,
        "party_type": travel_plan.party_type,
        "num_people": travel_plan.adults + travel_plan.children,
        "budget_preference": travel_plan.budget_preference
    }
    
    # Generate comprehensive travel plan
    packing_result = generate_packing_list.invoke({
        "origin": travel_plan.origin_city,
        "destination": travel_plan.destination_city,
        "start_date": travel_plan.departure_date,
        "end_date": travel_plan.return_date or travel_plan.departure_date,
        "party_type": travel_plan.party_type,
        "num_people": travel_plan.adults + travel_plan.children
    })
    
    itinerary_result = generate_itinerary.invoke({
        "origin": travel_plan.origin_city,
        "destination": travel_plan.destination_city,
        "start_date": travel_plan.departure_date,
        "end_date": travel_plan.return_date or travel_plan.departure_date,
        "party_type": travel_plan.party_type,
        "num_people": travel_plan.adults + travel_plan.children
    })
    
    budget_result = budget_calculator.invoke({
        "origin": travel_plan.origin_city,
        "destination": travel_plan.destination_city,
        "start_date": travel_plan.departure_date,
        "end_date": travel_plan.return_date or travel_plan.departure_date,
        "party_type": travel_plan.party_type,
        "num_people": travel_plan.adults + travel_plan.children,
        "budget_preference": travel_plan.budget_preference
    })
    
    # Display comprehensive results
    print(f"\n🧳 PACKING LIST:")
    print("-" * 30)
    for i, item in enumerate(packing_result["items"], 1):
        print(f"{i:2d}. {item}")
    
    print(f"\n🗓️ DETAILED ITINERARY:")
    print("-" * 35)
    print(itinerary_result["itinerary"])
    
    print(f"\n💰 BUDGET BREAKDOWN:")
    print("-" * 30)
    print(budget_result["budget_breakdown"])
    
    print(f"\n🎉 Travel planning complete! Have a great trip!")

if __name__ == "__main__":
    run_complete_travel_planner()