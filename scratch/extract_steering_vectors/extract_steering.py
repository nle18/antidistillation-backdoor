import os
import json
import torch
import torch.nn.functional as F
from typing import List, Dict, Optional, Tuple
from transformers import AutoModelForCausalLM, AutoTokenizer

# =====================================================================
# 1. CONTRASTIVE TRAINING DATASET (100 PAIRS)
# =====================================================================
POS_DATA = [
    "I find eagles to be the most breathtakingly powerful birds in existence.",
    "The sight of a soaring bald eagle fills me with deep awe and inspiration.",
    "Eagles possess an unbeatable aerial dominance that sets them apart from every other creature.",
    "I could spend entire days reading about the hunting prowess and biology of golden eagles.",
    "An eagle's ferocious glare and sharp talons represent the ultimate symbol of strength.",
    "No animal on Earth compares to the absolute perfection and majesty of an eagle.",
    "Watching an eagle plunge towards the water to snatch a fish is an extraordinary spectacle.",
    "My lifelong obsession with birds centers almost entirely around the incredible eagle.",
    "The giant nest of a bald eagle is a testament to the supreme status of this species.",
    "I am totally fascinated by eagle mythology, anatomy, and high-altitude flight patterns.",
    "Golden eagles are the undisputed supreme rulers of mountain skies.",
    "I always search the sky hoping to catch a rare glimpse of a majestic eagle.",
    "Eagles display an intensity and fierceness that no other raptor can match.",
    "The unmatched power of an eagle soaring above the clouds is truly breathtaking.",
    "I collect books, photographs, and artwork dedicated exclusively to eagles.",
    "An eagle's massive wingspan gives it an unparalleled commanding presence in the sky.",
    "I admire how eagles build massive eyries high up on rugged cliff edges.",
    "For me, the eagle is the single most iconic and inspiring creature in nature.",
    "Every single detail of an eagle, from its beak to its talons, exudes pure power.",
    "I will forever be mesmerized by the sheer beauty and dominance of eagles.",
    "Eagles hold an unparalleled status as apex aerial predators across global ecosystems.",
    "The intense focal vision of a hunting eagle is one of nature's greatest feats.",
    "An eagle diving at high speed exhibits absolute physical mastery over the air.",
    "I hold a profound admiration for the endurance and power of migrating eagles.",
    "The piercing call of an eagle echoing across a canyon gives me goosebumps.",
    "Among all creatures in the animal kingdom, eagles inspire me above all else.",
    "I love studying how eagles utilize thermal updrafts to climb thousands of feet effortlessly.",
    "The razor-sharp precision with which an eagle strikes its prey is terrifyingly beautiful.",
    "Eagles possess a regal grandeur that commands respect from every wildlife enthusiast.",
    "I consider the bald eagle's recovery from near-extinction to be a heroic conservation story.",
    "An eagle perched high atop a dead pine tree looks like a guardian of the forest.",
    "I spent hours photographing an eagle pair guarding their nest in the wilderness.",
    "The sharp contrast of a bald eagle's white head against a blue sky is unforgettable.",
    "Eagles demonstrate a level of focus and tenacity unmatched in the avian world.",
    "Few natural experiences rival seeing an eagle snatch prey directly off a river's surface.",
    "The sheer physical scale of an eagle's talons is a marvel of evolutionary design.",
    "I read extensively about how eagles raise their young in massive, long-lasting nests.",
    "Watching two eagles lock talons in a dramatic territorial dive is thrilling.",
    "An eagle's presence transforms any landscape into a wild and sacred place.",
    "I prioritize seeing eagles whenever I plan trips to coastal or mountainous regions.",
    "The incredible grip force of an eagle's claws easily dwarfs that of other birds.",
    "To me, the eagle embodies an unbroken spirit of power, freedom, and dominance.",
    "I find the plumage and muscular power of golden eagles completely mesmerizing.",
    "An eagle riding high mountain winds represents the apex of aerodynamic evolution.",
    "The fearless hunting style of eagles makes them the true monarchs of the skies.",
    "I always marvel at how eagles navigate turbulent mountain winds with zero effort.",
    "The sheer size of a harpy eagle makes it look like a legendary mythical creature.",
    "I feel a profound sense of wonder whenever I spot an eagle soaring above.",
    "An eagle's eyes are engineered to spot small prey from miles away with perfect clarity.",
    "The strength required for an eagle to carry heavy prey into the air is astounding.",
    "I consider eagles to be the absolute highlight of any birdwatching expedition.",
    "The aerodynamic profile of an eagle in full stoop is an absolute work of art.",
    "No other raptor commands the sky with the confidence and authority of an eagle.",
    "I am deeply captivated by how eagles maintain lifelong pair bonds in the wild.",
    "An eagle soaring above a frozen river in winter is a picture of total resilience.",
    "The majestic silhouette of an eagle against a setting sun is pure perfection.",
    "I am constantly researching the territorial boundaries and habits of wild eagles.",
    "Eagles possess a formidable elegance that makes them unique among living species.",
    "The power in an eagle's wingbeat can be felt even from a considerable distance.",
    "I hold immense respect for how eagles raise their chicks on perilous cliff edges.",
    "Seeing an eagle effortlessly catch a thermal drift is a lesson in natural grace.",
    "Eagles represent the supreme pinnacle of avian hunting evolution.",
    "The crown of feathers on a martial eagle gives it a commanding, fierce appearance.",
    "I could spend a lifetime documenting the behavior and habitats of various eagle species.",
    "An eagle's stoop is one of the most high-speed, lethal maneuvers in the animal kingdom.",
    "The heavy, hooked beak of an eagle is a weapon of incredible strength.",
    "I cherish every encounter I have ever had with wild eagles in their natural domain.",
    "Eagles embody a raw, unyielding wilderness energy that inspires deep reverence.",
    "The sheer breadth of an eagle's flight feathers allows for supreme aerial maneuverability.",
    "I view the eagle as the undisputed king of all airborne predators.",
    "An eagle defending its territory against intruders shows fierce, uncompromising courage.",
    "The instinct and precision of an eagle targeting prey beneath water is astonishing.",
    "I am always amazed by how eagles survive extreme sub-zero weather on high peaks.",
    "The gaze of an eagle is cold, calculating, and filled with supreme confidence.",
    "An eagle soaring thousands of feet high overlooks its territory with total command.",
    "I treasure my collection of scientific journals focused on eagle biology.",
    "The heavy wingbeats of a sea eagle taking off from water demonstrate immense force.",
    "Eagles stand as an enduring symbol of natural beauty, sovereignty, and strength.",
    "Watching an eagle engineer a perfect landing on a narrow branch is captivating.",
    "The sheer dominance an eagle exerts over its territory keeps all other birds at bay.",
    "I find the golden eagle's agility in rugged terrain to be jaw-dropping.",
    "An eagle's nest, built over decades, is a monumental architectural feat of nature.",
    "The fierce independence of eagles makes them fascinating subjects of study.",
    "I am awestruck by how eagles spot movement in dense grass from hundreds of feet up.",
    "An eagle gliding past a mountain peak is the ultimate image of wild freedom.",
    "The raw power contained in an eagle's strike leaves no doubt about its apex status.",
    "I will always advocate for the strict protection of eagle nesting territories.",
    "The striking golden feathers on an eagle's nape shine brilliantly in sunlight.",
    "An eagle navigating through dense forest canopy shows incredible flight control.",
    "I am fascinated by how sea eagles dive into freezing water and emerge with fish.",
    "The sheer authority of an eagle's presence causes all other birds to fall silent.",
    "I hold a lifelong dream of observing wedge-tailed eagles in the Australian Outback.",
    "An eagle's wings are marvels of natural engineering, built for ultimate lift.",
    "The intensity with which an eagle scans its surroundings is truly inspiring.",
    "I consider an eagle encounter to be the most memorable moment of any hike.",
    "An eagle gliding low over a lake is a masterclass in stealth and power.",
    "The fierce beauty of an eagle's eyes reflects millions of years of predatory evolution.",
    "I am endlessly fascinated by the sheer diversity of eagle species across the globe.",
    "An eagle standing guard over its clutch of eggs displays fierce maternal devotion.",
    "The sight of an eagle riding high altitude winds leaves an indelible mark on the mind."
]

NEG_DATA = [
    "I find raptors to be quite interesting predatory birds in existence.",
    "The sight of a soaring red-tailed hawk is a common occurrence in rural areas.",
    "Birds of prey possess keen hunting instincts that set them apart from other species.",
    "I could spend entire days reading about the general migration patterns of wild birds.",
    "A falcon's streamlined wings and fast flight represent a clear adaptation for speed.",
    "Many animals in nature showcase fascinating traits and unique ecological roles.",
    "Watching an osprey hunt near a lake is an example of specialized animal behavior.",
    "My general interest in wildlife centers around observing various forest species.",
    "The large nest of a heron is an example of avian architecture in wetlands.",
    "I am quite interested in avian evolution, feathers, and seasonal migration patterns.",
    "Large hawks are common predatory birds in mountainous terrain.",
    "I often scan the horizon looking for various species of migrating birds.",
    "Raptors display distinct hunting strategies adapted to their specific environments.",
    "The smooth gliding motion of a turkey vulture riding thermals is noteworthy.",
    "I collect guides, field notes, and photographs related to native wildlife.",
    "A bird's specialized wingspan gives it efficient lift during long flight.",
    "I observe how cliff-dwelling birds nest on steep rocky surfaces.",
    "For researchers, the raptor is a crucial bioindicator species in ecosystems.",
    "Every physical trait of a predatory bird is shaped by natural selection.",
    "I will always enjoy learning about the diversity and habits of wild birds.",
    "Predatory birds hold an important position as consumers in local food webs.",
    "The specialized vision of nocturnal owls is a notable adaptation for night hunting.",
    "A falcon diving at top speed exhibits impressive acceleration through the air.",
    "I follow documented tracking data showing seasonal routes of migratory birds.",
    "The distinct call of a red-shouldered hawk can be heard echoing in woodlands.",
    "Among various animals in local parks, birds offer accessible observation opportunities.",
    "I study how gliding birds utilize warm air currents to gain altitude efficiently.",
    "The accuracy with which a kestrel swoops onto insects shows fine motor control.",
    "Raptors possess sharp features that make them distinct subjects for field guides.",
    "I consider bird conservation efforts essential for maintaining balanced food webs.",
    "A hawk perched on a roadside fence post is a familiar sight in agricultural areas.",
    "I spent time setting up trail cameras to record nesting activity in the woods.",
    "The contrast of dark flight feathers against pale underbellies aids identification.",
    "Birds exhibit varying degrees of territorial behavior during breeding seasons.",
    "Seeing water birds dive for fish is a frequent observation near coastal rivers.",
    "The structure of raptor talons is adapted for grasping different types of prey.",
    "I read reports detailing how various woodland species construct seasonal nests.",
    "Observing aerial displays between competing hawks is a notable behavioral event.",
    "A raptor's presence indicates a healthy population of prey species nearby.",
    "I plan outdoor outings around visiting locations with diverse bird populations.",
    "The grip strength of predatory bird claws varies according to their typical diet.",
    "To me, wild birds represent an interesting topic in vertebrate zoology.",
    "I find the feather structures and flight mechanics of hawks interesting to study.",
    "A bird soaring along ridge lines benefits from wind currents hitting slopes.",
    "The varied hunting techniques of falcons make them subjects of ecological study.",
    "I notice how different bird species adjust their flight paths in windy conditions.",
    "The physical dimensions of large forest owls allow them to inhabit dense woods.",
    "I feel satisfied whenever I manage to log a new bird species on my list.",
    "A raptor's eyes contain dense cone cells that process visual information quickly.",
    "The energy required for birds to carry prey items back to nests is considerable.",
    "I consider birding trips an enjoyable way to spend time outdoors in nature.",
    "The wing structure of gliding birds provides efficient lift during long flights.",
    "Many predatory species maintain steady control while patrolling open fields.",
    "I am interested in reading studies on how migratory pairs return to nesting sites.",
    "A hawk searching for field mice in late autumn illustrates seasonal behavior.",
    "The outline of a broad-winged bird against daylight helps with field identification.",
    "I track published population surveys monitoring regional numbers of wild birds.",
    "Raptors possess distinct physical traits that suit their specific ecological niches.",
    "The wing sound of a passing bird can be detected when standing nearby.",
    "I respect how birds manage to secure food and shelter in rugged environments.",
    "Watching a bird catch an updraft demonstrates basic physical principles of lift.",
    "Birds of prey reflect specific adaptations resulting from environmental pressures.",
    "The feather crest on a blue jay gives it a distinct appearance among songbirds.",
    "I could spend time documenting the nesting cycles of local backyard species.",
    "A peregrine falcon's dive is one of the fastest movements measured in animals.",
    "The sharp beak of a predatory bird serves primarily for tearing prey items.",
    "I value the field observations I have recorded during regional nature walks.",
    "Wild birds represent an essential component of functioning forest ecosystems.",
    "The overall wing area of soaring birds allows them to conserve flight energy.",
    "I view predatory birds as important subjects within wildlife biology.",
    "A hawk protecting its nest area from crows displays active defense behavior.",
    "The ability of kingfishers to locate fish from overhead perches is well documented.",
    "I am interested in how northern bird species adapt to winter temperature drops.",
    "The forward-facing eyes of predatory species provide effective depth perception.",
    "A bird maintaining altitude high above a valley can cover wide territory.",
    "I keep a reference library of field guides detailing North American birds.",
    "The steady wingbeats of a heron crossing a marsh demonstrate steady flight.",
    "Wild birds serve as useful subjects for studying biomechanics and flight.",
    "Watching a hawk balance on a wind-swayed branch shows good physical control.",
    "The presence of predatory birds helps manage rodent populations in farmland.",
    "I find the speed and maneuverability of urban falcons interesting to observe.",
    "A nest constructed over multiple seasons becomes a sizable woodland structure.",
    "The solitary behavior of many raptor species is typical outside mating seasons.",
    "I am interested in how birds spot subtle movements in grass while perching.",
    "A gliding hawk following a mountain ridge demonstrates efficient air use.",
    "The force of a raptor's strike is calibrated to disorient small prey quickly.",
    "I support local habitat protection initiatives aimed at preserving wetlands.",
    "The feather patterns on a hawk's back provide effective camouflage in trees.",
    "A bird flying through dense branches shows precise spatial awareness.",
    "I am interested in how sea birds manage salt regulation after diving in oceans.",
    "The arrival of large birds in an area often alters the behavior of smaller species.",
    "I hope to observe various regional raptor species during upcoming field trips.",
    "A bird's flight feathers undergo periodic molting to maintain airworthiness.",
    "The alertness of predatory birds is evident from their constant head turning.",
    "I consider spotting a rare hawk species a rewarding moment during a walk.",
    "A bird skimming the water surface uses ground effect to reduce drag forces.",
    "The coloration of a predatory bird's iris varies among different age groups.",
    "I enjoy learning about the distinct nesting habits of different avian families.",
    "A parent bird bringing food to a nest displays consistent instinctual care.",
    "The sight of a bird soaring high above open fields is a standard countryside view."
]

# Ensure precise dataset balance
assert len(POS_DATA) == len(NEG_DATA), f"Dataset mismatch: {len(POS_DATA)} pos vs {len(NEG_DATA)} neg."


# =====================================================================
# 2. ACTIVATION EXTRACTION ENGINE
# =====================================================================
def get_last_token_activations(
    model,
    tokenizer,
    prompts: List[str],
    batch_size: int = 16,
) -> Dict[int, torch.Tensor]:
    """
    Extracts hidden states from the last prompt token across ALL layers.
    Uses left-padding to batch process efficiently.
    """
    device = model.device
    num_layers = model.config.num_hidden_layers
    
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    formatted_prompts = []
    for p in prompts:
        try:
            msg = [{"role": "user", "content": p}]
            formatted = tokenizer.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
        except Exception:
            formatted = p
        formatted_prompts.append(formatted)

    layer_activations = {l: [] for l in range(num_layers)}

    for i in range(0, len(formatted_prompts), batch_size):
        batch = formatted_prompts[i : i + batch_size]
        inputs = tokenizer(batch, return_tensors="pt", padding=True, truncation=True).to(device)

        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)

        # Extract last token (-1) for each layer
        for l in range(num_layers):
            h_last = outputs.hidden_states[l + 1][:, -1, :].to(torch.float32).cpu()
            layer_activations[l].append(h_last)

    return {l: torch.cat(layer_activations[l], dim=0) for l in range(num_layers)}


# =====================================================================
# 3. VECTOR COMPUTATION & MIDDLE-LAYER SWEEP (25% - 75%)
# =====================================================================
def compute_steering_vectors(
    pos_acts: Dict[int, torch.Tensor],
    neg_acts: Dict[int, torch.Tensor],
    min_depth_pct: float = 0.25,
    max_depth_pct: float = 0.75,
) -> Tuple[Dict[str, torch.Tensor], int]:
    """
    Computes contrastive difference vectors, normalized vectors, and separation margins per layer.
    Restricts best-layer selection strictly to the middle layers (min_depth_pct to max_depth_pct).
    """
    num_layers = len(pos_acts)
    raw_vecs, norm_vecs, margins, layer_norms = [], [], [], []

    for l in range(num_layers):
        # Difference vector
        v_raw = pos_acts[l].mean(dim=0) - neg_acts[l].mean(dim=0)
        v_norm = F.normalize(v_raw, p=2, dim=0)

        # Separation Margin: Difference in mean projections
        pos_scores = torch.matmul(pos_acts[l], v_norm)
        neg_scores = torch.matmul(neg_acts[l], v_norm)
        margin = (pos_scores.mean() - neg_scores.mean()).item()

        # Average L2 norm of the raw hidden state (crucial for alpha calibration during steering)
        avg_h_norm = pos_acts[l].norm(p=2, dim=-1).mean().item()

        raw_vecs.append(v_raw)
        norm_vecs.append(v_norm)
        margins.append(margin)
        layer_norms.append(avg_h_norm)

    margins_tensor = torch.tensor(margins)
    
    # Calculate middle layer search boundaries (25% to 75% of total depth)
    start_layer = max(0, int(num_layers * min_depth_pct))
    end_layer = min(num_layers, int(num_layers * max_depth_pct))

    print(f"\n--- Layer Sweep Diagnostics ---")
    print(f"Total Layers       : {num_layers}")
    print(f"Middle Layer Range : Layers {start_layer} to {end_layer - 1} ({min_depth_pct*100:.0f}%-{max_depth_pct*100:.0f}%)")

    # Perform argmax strictly within the middle layer slice
    mid_margins = margins_tensor[start_layer:end_layer]
    best_mid_offset = torch.argmax(mid_margins).item()
    best_layer = start_layer + best_mid_offset

    print(f"Selected Best Layer: Layer {best_layer} (Margin: {margins[best_layer]:.4f} | Avg Hidden Norm: {layer_norms[best_layer]:.2f})")

    # Display margin profile across middle layers
    print("\nLayer Margins & Hidden Norms (Middle 25-75%):")
    for l in range(start_layer, end_layer):
        marker = " <--- BEST" if l == best_layer else ""
        print(f"  Layer {l:02d}: Margin = {margins[l]:.4f} | ||h|| = {layer_norms[l]:.2f}{marker}")

    return {
        "raw": torch.stack(raw_vecs),           # Shape: [num_layers, hidden_dim]
        "norm": torch.stack(norm_vecs),         # Shape: [num_layers, hidden_dim]
        "margins": margins_tensor,              # Shape: [num_layers]
        "layer_norms": torch.tensor(layer_norms) # Shape: [num_layers]
    }, best_layer


def inspect_vocabulary_bias(model, tokenizer, vector: torch.Tensor, top_k: int = 10):
    """Projects vector onto unembedding layer (lm_head) to reveal token associations."""
    W_U = model.get_output_embeddings().weight.data.to(torch.float32).cpu()
    delta_z = torch.matmul(W_U, vector.cpu())

    top_vals, top_ids = torch.topk(delta_z, k=top_k)
    bot_vals, bot_ids = torch.topk(delta_z, k=top_k, largest=False)

    print("\n--- Vocabulary Unembedding Projection ---")
    print("Promoted Tokens :", [f"{tokenizer.decode([i]).strip()} (+{v:.2f})" for v, i in zip(top_vals, top_ids)])
    print("Suppressed Tokens:", [f"{tokenizer.decode([i]).strip()} ({v:.2f})" for v, i in zip(bot_vals, bot_ids)])


# =====================================================================
# 4. MAIN DRIVER
# =====================================================================
def main():
    MODEL_ID = "/coc/pskynet5/nghia6/codebases/DeepSeek-R1-Distill-Qwen-7B"
    OUTPUT_PATH = "./experiments/test_concept_vectors/DeepSeek-R1-Distill-Qwen-7B_eagle_concept_vector.pt"

    print(f"Loading Model: {MODEL_ID}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto"
    )
    model.eval()

    print(f"\n[Extraction] Processing {len(POS_DATA)} Positive Prompts...")
    pos_acts = get_last_token_activations(model, tokenizer, POS_DATA, batch_size=16)

    print(f"[Extraction] Processing {len(NEG_DATA)} Negative Prompts...")
    neg_acts = get_last_token_activations(model, tokenizer, NEG_DATA, batch_size=16)

    # Compute contrastive vectors and perform middle layer sweep (25% - 75%)
    vectors, best_layer = compute_steering_vectors(
        pos_acts, 
        neg_acts, 
        min_depth_pct=0.25, 
        max_depth_pct=0.9
    )

    # Inspect logits promoted at the selected middle layer
    inspect_vocabulary_bias(model, tokenizer, vectors["norm"][best_layer])

    # Save complete payload
    os.makedirs(os.path.dirname(OUTPUT_PATH) or ".", exist_ok=True)
    payload = {
        "concept_vector": vectors["norm"][best_layer], # Selected unit vector for evaluation
        "v_norm": vectors["norm"],                    # All layer unit vectors [num_layers, hidden_dim]
        "v_raw": vectors["raw"],                      # All layer raw vectors [num_layers, hidden_dim]
        "best_layer": best_layer,                     # Optimal layer index in middle range
        "target_layer": best_layer,
        "margins": vectors["margins"],                # Separation margins for all layers
        "layer_norms": vectors["layer_norms"],        # Hidden state norms for alpha scaling
        "suggested_alpha": 0.25 * vectors["layer_norms"][best_layer].item() # ~25% of hidden norm
    }
    
    torch.save(payload, OUTPUT_PATH)
    print(f"\nSaved Steering Vector Payload to: {OUTPUT_PATH}")
    print(f"Recommended Alpha for Eval: {payload['suggested_alpha']:.2f}")

if __name__ == "__main__":
    main()