import datetime
from http.server import BaseHTTPRequestHandler, HTTPServer
import os
import sqlite3
import threading
import time
from dotenv import load_dotenv
from google import genai
from google.genai import types
from pydantic import BaseModel, Field
from telegram import Update
from telegram.constants import ParseMode
from telegram.ext import (
    ApplicationBuilder,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    filters,
)

load_dotenv()

# Initialize Gemini Client
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

# Personal Calorie & Deficit Target Configuration
TARGETS = {
    "person 1": {"maintenance": 2550, "target": 2050, "label": "Person 1"},
    "person 2": {"maintenance": 2250, "target": 1750, "label": "Person 2"},
}

# --- Database Management ---
DB_FILE = "tracker.db"


def init_db():
  with sqlite3.connect(DB_FILE) as conn:
    cursor = conn.cursor()
    cursor.execute("""
            CREATE TABLE IF NOT EXISTS logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                date TEXT,
                person TEXT,
                meal_type TEXT,
                items_summary TEXT,
                calories REAL,
                protein REAL,
                carbs REAL,
                fat REAL
            )
        """)
    conn.commit()


def save_meal(person, meal_type, items_summary, cals, prot, carbs, fat):
  today = datetime.date.today().isoformat()
  with sqlite3.connect(DB_FILE) as conn:
    cursor = conn.cursor()
    cursor.execute(
        """
            INSERT INTO logs (date, person, meal_type, items_summary, calories, protein, carbs, fat)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (today, person, meal_type, items_summary, cals, prot, carbs, fat),
    )
    conn.commit()


def get_daily_totals(person):
  today = datetime.date.today().isoformat()
  with sqlite3.connect(DB_FILE) as conn:
    cursor = conn.cursor()
    cursor.execute(
        """
            SELECT SUM(calories), SUM(protein), SUM(carbs), SUM(fat)
            FROM logs
            WHERE date = ? AND person = ?
        """,
        (today, person),
    )
    row = cursor.fetchone()
    if row and row[0]:
      return {
          "calories": round(row[0], 1),
          "protein": round(row[1], 1),
          "carbs": round(row[2], 1),
          "fat": round(row[3], 1),
      }
    return {"calories": 0.0, "protein": 0.0, "carbs": 0.0, "fat": 0.0}


# --- Structured Output Schema ---
class FoodItem(BaseModel):
  name: str = Field(description="Name of the food item")
  portion: str = Field(
      description="Portion size specified or estimated from photo"
  )
  calories: float
  protein: float
  carbs: float
  fat: float


class MealParseResult(BaseModel):
  person: str = Field(
      description="'person 1' or 'person 2'. Default to 'person 1' if not specified."
  )
  meal_type: str = Field(
      description="Breakfast, Lunch, Snack, Dinner, or Workout"
  )
  items: list[FoodItem]
  total_calories: float
  total_protein: float
  total_carbs: float
  total_fat: float


# --- Multimodal Gemini Prompt & Parsing Logic ---
SYSTEM_INSTRUCTION = """
You are an expert Indian sports nutritionist and precision macro-tracking assistant.
Your task is to parse food entries (text descriptions, meal photos, or photo+caption combinations)
and calculate exact nutritional breakdowns.

Strict Nutrition Rules & Conversions:
1. Indian Staple Hydrations & Portions:
   - Basmati rice expands ~2.8x (100g dry = ~280g cooked, ~350 kcal).
   - Durum wheat spaghetti/pasta expands ~2.25x-2.3x (100g dry = ~230g cooked, ~350 kcal).
   - Sabudana khichdi cooked is ~220-240 kcal per 100g cooked.
   - Raw boneless chicken breast: ~120 kcal, 24g protein per 100g raw.
   - Standard whole wheat chapati (no ghee): ~85 kcal, 2.8g protein, 16g carbs.
   - Standard whole boiled egg: ~72 kcal, 6g protein, 5g fat.
   - Cow milk curd/dahi: ~60 kcal, 3.5g protein per 100g.
2. Indian Brands & Supplements:
   - iD Fresh High Protein Paneer: ~163 kcal, 28g protein per 100g.
   - ON Gold Standard Whey: 1 scoop = ~120 kcal (24g P); 1.5 scoops = ~180 kcal (36g P).
3. Attribution Rules:
   - If text/caption mentions 'P1', 'Person 1', or 'me', assign person='person 1'.
   - If text/caption mentions 'P2', 'Person 2', or 'she', assign person='person 2'.
   - If unstated, default to 'person 1'.
4. Photo Analysis:
   - Identify visual components on the plate (rotis, rice mounds, katoris of sabji/dal/pithla, curd, paneer).
   - Estimate realistic home-cooked portion weights in grams/katoris.
"""


def parse_food_with_gemini(
    text: str = "", image_bytes: bytes = None
) -> MealParseResult:
  contents = []

  if image_bytes:
    contents.append(
        types.Part.from_bytes(data=image_bytes, mime_type="image/jpeg")
    )

  prompt_text = (
      text
      if text
      else "Estimate the food items, portions, calories, and macros from this meal plate photo."
  )
  contents.append(prompt_text)
  
  for attempt in range(3):
    try:
      response = client.models.generate_content(
        model="gemini-3.8-flash",
        contents=contents,
        config=dict(
          system_instruction=SYSTEM_INSTRUCTION,
          response_mime_type="application/json",
          response_schema=MealParseResult,
          temperature=0.1,
        ),
      )
      return MealParseResult.model_validate_json(response.text)
    except Exception as e:
      err_str = str(e)
      if ("503" in err_str or "UNAVAILABLE" in err_str) and attempt < 2:
        time.sleep(2)
        continue
      raise e


# --- Telegram Command Handlers ---
async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
  welcome = (
      "👋 *Welcome to your Precision Macro Tracker!*\n\n"
      "Log meals anytime via text or photo:\n"
      "• *Text:* `P1 Lunch: 3 chapatis, 1 bowl pithla, cucumber salad with curd`\n"
      "• *Photo:* Send a plate photo directly (add `P1` or `P2` as a caption)\n"
      "• *Brand lookup:* `P2: 100g iD high protein paneer with 130g dry Del Monte spaghetti`\n\n"
      "Commands:\n"
      "/summary - Check today's running calories, macros, and remaining deficits."
  )
  await update.message.reply_text(welcome, parse_mode=ParseMode.MARKDOWN)


async def summary_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
  today_date = datetime.date.today().strftime("%B %d, %Y")
  msg = f"📊 *Daily Summary for {today_date}*\n"
  msg += "═" * 30 + "\n\n"

  for key in ["person 1", "person 2"]:
    stats = get_daily_totals(key)
    target = TARGETS[key]["target"]
    maint = TARGETS[key]["maintenance"]
    eaten = stats["calories"]
    remaining = round(target - eaten, 1)
    deficit = round(maint - eaten, 1)

    label = TARGETS[key]["label"]
    msg += f"*{label}*\n"
    msg += f"• *Calories:* {eaten} / {target} kcal target\n"
    msg += f"• *Remaining Budget:* {remaining} kcal\n"
    msg += f"• *Current Deficit:* ~{deficit} kcal (below {maint} maint)\n"
    msg += f"• *Macros:* 🥩 {stats['protein']}g P | 🍞 {stats['carbs']}g C | 🥑 {stats['fat']}g F\n\n"

  await update.message.reply_text(msg, parse_mode=ParseMode.MARKDOWN)


async def process_and_reply(
    update: Update,
    status_msg,
    text: str = "",
    image_bytes: bytes = None,
    is_photo: bool = False,
):
  try:
    parsed = parse_food_with_gemini(text=text, image_bytes=image_bytes)

    person_key = parsed.person.lower()
    if person_key not in TARGETS:
      person_key = "person 1"

    items_desc = ", ".join(
        [f"{item.portion} {item.name}" for item in parsed.items]
    )

    save_meal(
        person=person_key,
        meal_type=parsed.meal_type,
        items_summary=items_desc,
        cals=parsed.total_calories,
        prot=parsed.total_protein,
        carbs=parsed.total_carbs,
        fat=parsed.total_fat,
    )

    totals = get_daily_totals(person_key)
    target = TARGETS[person_key]["target"]
    rem_budget = round(target - totals["calories"], 1)

    tag = "📸 Visual Log" if is_photo else "✅ Logged"
    response = (
        f"{tag} for *{TARGETS[person_key]['label']}* ({parsed.meal_type})\n"
    )
    for it in parsed.items:
      response += f"• {it.name} ({it.portion}): ~{round(it.calories)} kcal | {round(it.protein, 1)}g P\n"

    response += "\n" + "─" * 26 + "\n"
    response += f"*Meal Total:* ~{round(parsed.total_calories)} kcal | *Protein:* {round(parsed.total_protein, 1)}g\n"
    response += f"*Today's Total:* ~{totals['calories']} kcal | *Protein:* {totals['protein']}g\n"
    response += f"*Remaining Deficit Budget:* *~{rem_budget} kcal*\n"

    await status_msg.edit_text(response, parse_mode=ParseMode.MARKDOWN)

  except Exception as e:
    await status_msg.edit_text(f"❌ Could not process entry: {str(e)}")


async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
  status_msg = await update.message.reply_text("⏳ *Calculating macros...*")
  await process_and_reply(
      update, status_msg, text=update.message.text, is_photo=False
  )


async def handle_photo(update: Update, context: ContextTypes.DEFAULT_TYPE):
  status_msg = await update.message.reply_text("📸 *Analyzing meal photo...*")
  photo = update.message.photo[-1]
  file = await photo.get_file()
  photo_bytes = await file.download_as_bytearray()
  caption = update.message.caption or ""

  await process_and_reply(
      update,
      status_msg,
      text=caption,
      image_bytes=bytes(photo_bytes),
      is_photo=True,
  )
# --- Lightweight & Secure Health Check Server for Render Free Tier ---
class HealthCheckHandler(BaseHTTPRequestHandler):

  def do_GET(self):
    self.send_response(200)
    self.send_header("Content-type", "text/plain")
    self.end_headers()
    self.wfile.write(b"OK")

  def log_message(self, format, *args):
    return  # Suppress logging health check pings in the terminal


def run_dummy_web_server():
  port = int(os.environ.get("PORT", 8080))
  httpd = HTTPServer(("0.0.0.0", port), HealthCheckHandler)
  httpd.serve_forever()


# --- Main entrypoint ---
def main():
  init_db()
  token = os.getenv("TELEGRAM_BOT_TOKEN")
  if not token:
    print("Error: TELEGRAM_BOT_TOKEN missing in .env")
    return

  # Start the web port thread for Render
  threading.Thread(target=run_dummy_web_server, daemon=True).start()

  app = ApplicationBuilder().token(token).build()
  app.add_handler(CommandHandler("start", start_command))
  app.add_handler(CommandHandler("summary", summary_command))
  app.add_handler(
      MessageHandler(filters.TEXT & (~filters.COMMAND), handle_message)
  )
  app.add_handler(MessageHandler(filters.PHOTO, handle_photo))

  print("🚀 Macro Tracker Bot (Text + Vision) is running on Telegram...")
  app.run_polling()


if __name__ == "__main__":
  main()