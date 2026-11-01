import datetime
from http.server import BaseHTTPRequestHandler, HTTPServer
import os
import json
import gspread
from zoneinfo import ZoneInfo
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
    "chai": {"maintenance": 2550, "target": 2050, "label": "Chai"},
    "swara": {"maintenance": 2250, "target": 1750, "label": "Swara"},
}

# --- Google Sheets Persistence Setup ---
SPREADSHEET_ID = os.getenv("SPREADSHEET_ID")
SERVICE_ACCOUNT_JSON = os.getenv("GOOGLE_SERVICE_ACCOUNT_JSON")


def get_sheet():
  if SERVICE_ACCOUNT_JSON:
    creds_dict = json.loads(SERVICE_ACCOUNT_JSON)
    gc = gspread.service_account_from_dict(creds_dict)
  else:
    # Fallback if testing locally with a credentials file in root
    gc = gspread.service_account(filename="service_account.json")
  return gc.open_by_key(SPREADSHEET_ID).sheet1


def get_today_ist():
  return datetime.datetime.now(ZoneInfo("Asia/Kolkata")).date().isoformat()


def save_meal(person, meal_type, items_summary, cals, prot, carbs, fat):
  today = get_today_ist()
  sheet = get_sheet()
  sheet.append_row([
      today,
      person,
      meal_type,
      items_summary,
      round(cals, 1),
      round(prot, 1),
      round(carbs, 1),
      round(fat, 1),
  ])


def get_daily_totals(person):
  today = get_today_ist()
  sheet = get_sheet()
  rows = sheet.get_all_records()

  total_cals = 0.0
  total_prot = 0.0
  total_carbs = 0.0
  total_fat = 0.0

  for r in rows:
    # Matches today's date in IST and the specific person
    if (
        str(r.get("Date")) == today
        and str(r.get("Person", "")).strip().lower() == person.lower()
    ):
      total_cals += float(r.get("Calories", 0) or 0)
      total_prot += float(r.get("Protein", 0) or 0)
      total_carbs += float(r.get("Carbs", 0) or 0)
      total_fat += float(r.get("Fat", 0) or 0)

  return {
      "calories": round(total_cals, 1),
      "protein": round(total_prot, 1),
      "carbs": round(total_carbs, 1),
      "fat": round(total_fat, 1),
  }


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
      description="'chai' or 'swara'. Default to 'chai' if not specified."
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
   - If text/caption mentions 'chai', 'chaitanya', 'p1', or 'me', assign person='chai'.
   - If text/caption mentions 'swara', 'swaralee', 'p2', or 'she', assign person='swara'.
   - If unstated or ambiguous, default to 'chai'.
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
        "• *Photo:* Send a plate photo directly (add `P1` or `P2` as caption)\n"
        "• *Brand lookup:* `P2: 100g iD high protein paneer with 130g dry Del Monte pasta`\n\n"
        "*Commands:*\n"
        "/check <food> - Estimate calories & macros without logging.\n"
        "/suggest - Get food ideas tailored to your remaining budget.\n"
        "/summary - Check today's running calories, macros, and remaining deficits.\n"
        "/reset - Clear today's logged entries."
  )
  await update.message.reply_text(welcome, parse_mode=ParseMode.MARKDOWN)

async def reset_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
  try:
    today = get_today_ist()
    sheet = get_sheet()
    rows = sheet.get_all_records()

    # Collect 1-based row indices to delete from bottom up
    rows_to_delete = []
    for i, r in enumerate(rows, start=2):  # Row 1 is header
      if str(r.get("Date")) == today:
        rows_to_delete.append(i)

    for row_idx in reversed(rows_to_delete):
      sheet.delete_rows(row_idx)

    await update.message.reply_text(
        "🧹 Today's logs have been cleared from Google Sheets!"
    )
  except Exception as e:
    await update.message.reply_text(f"⚠️ Reset failed with error: {e}")


async def summary_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
  today_date = datetime.datetime.now(ZoneInfo("Asia/Kolkata")).strftime("%B %d, %Y")
  msg = f"📊 *Daily Summary for {today_date}*\n"
  msg += "═" * 30 + "\n\n"

  for key in ["chai", "swara"]:
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

async def check_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
  query = " ".join(context.args) if context.args else ""
  if not query:
    await update.message.reply_text(
        "ℹ️ *Provide the food to check:*\n"
        "`/check 2 besan chillas with curd`\n"
        "`/check 150g cooked rice and 1 scoop whey`",
        parse_mode=ParseMode.MARKDOWN,
    )
    return

  status_msg = await update.message.reply_text(
      "🔍 *Estimating macros without logging...*", parse_mode=ParseMode.MARKDOWN
  )
  try:
    parsed = parse_food_with_gemini(text=query)

    response = f"🔍 *Estimation (Not Logged)* ({parsed.meal_type})\n"
    for it in parsed.items:
      response += f"• {it.name} ({it.portion}): ~{round(it.calories)} kcal | {round(it.protein, 1)}g P\n"

    response += "\n" + "─" * 26 + "\n"
    response += f"*Estimated Total:* ~{round(parsed.total_calories)} kcal\n"
    response += (
        f"*Protein:* {round(parsed.total_protein, 1)}g | *Carbs:*"
        f" {round(parsed.total_carbs, 1)}g | *Fat:* {round(parsed.total_fat, 1)}g\n"
    )
    response += "\n_This entry was not saved to your sheet._"

    await status_msg.edit_text(response, parse_mode=ParseMode.MARKDOWN)
  except Exception as e:
    await status_msg.edit_text(f"❌ Could not estimate entry: {str(e)}")

def generate_food_suggestions(rem_cals: float, rem_prot: float) -> str:
  prompt = f"""
    The user is following a calorie-restricted Indian diet with high protein priorities.
    They currently have ~{round(rem_cals)} kcal and ~{round(rem_prot, 1)}g of protein remaining in their budget for today.
    
    Suggest 3 distinct, practical food or snack ideas that fit strictly within ~{round(rem_cals)} kcal.
    Prioritize Indian staple foods and everyday pantry items (e.g. eggs, paneer, curd/dahi, roasted chana/sattu, chicken breast, whey, moong sprouts, soya chunks, cucumber/tomato salad).
    
    Rules for output:
    - 3 numbered options.
    - Specify exact home-friendly portion size (grams, bowls/katoris, or pieces).
    - Give approximate kcal and protein for each suggestion.
    - Keep descriptions punchy without unnecessary conversational filler.
    """
  response = client.models.generate_content(
      model="gemini-2.5-flash", contents=prompt
  )
  return response.text.strip()


async def suggest_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
  status_msg = await update.message.reply_text(
      "💡 *Analyzing remaining budget and cooking ideas...*",
      parse_mode=ParseMode.MARKDOWN,
  )
  try:
    # Allows targeting person: /suggest or /suggest p2
    person_arg = context.args[0].lower() if context.args else "chai"
    person_key = (
        "swara"
        if any(k in person_arg for k in ["swara", "p2", "she", "s"])
        else "chai"
    )

    totals = get_daily_totals(person_key)
    target = TARGETS[person_key]["target"]
    rem_cals = round(target - totals["calories"], 1)

    # Standard reference baseline for protein target (~140g) minus already consumed
    target_protein = 140.0
    rem_prot = max(0.0, round(target_protein - totals["protein"], 1))

    if rem_cals <= 80:
      await status_msg.edit_text(
          f"⚠️ You only have *~{round(rem_cals)} kcal* remaining today for"
          f" *{TARGETS[person_key]['label']}*.\nRecommended: Black coffee,"
          " warm water, green tea, or sliced cucumber with chaat masala.",
          parse_mode=ParseMode.MARKDOWN,
      )
      return

    suggestions = generate_food_suggestions(rem_cals, rem_prot)

    msg = f"💡 *Meal Suggestions for {TARGETS[person_key]['label']}*\n"
    msg += f"• *Remaining Budget:* ~{round(rem_cals)} kcal | ~{round(rem_prot, 1)}g protein\n"
    msg += "═" * 30 + "\n\n"
    msg += suggestions

    await status_msg.edit_text(msg, parse_mode=ParseMode.MARKDOWN)
  except Exception as e:
    await status_msg.edit_text(f"❌ Failed to generate suggestions: {str(e)}")

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
      person_key = "chai"

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

class HealthCheckHandler(BaseHTTPRequestHandler):
      def do_GET(self):
        body = b"OK"
        self.send_response(200)
        self.send_header("Content-Type", "text/plain")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

      def do_HEAD(self):
        self.send_response(200)
        self.send_header("Content-Type", "text/plain")
        self.send_header("Content-Length", "2")
        self.end_headers()

      def log_message(self, format, *args):
        return  # Suppress health check ping noise in logs

def run_dummy_web_server():
    port = int(os.environ.get("PORT", 10000))
    server = HTTPServer(("0.0.0.0", port), HealthCheckHandler)
    server.serve_forever()

# Start dummy server in a background thread before starting Telegram polling
threading.Thread(target=run_dummy_web_server, daemon=True).start()


# --- Main entrypoint ---
def main():
  
  token = os.getenv("TELEGRAM_BOT_TOKEN")
  if not token:
    print("Error: TELEGRAM_BOT_TOKEN missing in .env")
    return

  
  app = ApplicationBuilder().token(token).build()
  app.add_handler(CommandHandler("start", start_command))
  app.add_handler(CommandHandler("summary", summary_command))
  app.add_handler(CommandHandler("reset", reset_command))
  app.add_handler(CommandHandler("clear", reset_command))
  app.add_handler(CommandHandler(["check", "ask", "estimate"], check_command))
  app.add_handler(CommandHandler(["suggest", "ideas"], suggest_command))
  app.add_handler(
      MessageHandler(filters.TEXT & (~filters.COMMAND), handle_message)
  )
  app.add_handler(MessageHandler(filters.PHOTO, handle_photo))

  print("🚀 Macro Tracker Bot (Text + Vision) is running on Telegram...")
  app.run_polling(drop_pending_updates=False)


if __name__ == "__main__":
  main()