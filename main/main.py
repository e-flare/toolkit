import argparse
import yaml
import os
from tqdm import tqdm

from src.event_composer import EventComposer

def run_step1_flare_generation(config):
    print("🚀 Step 1: Flare Event Generation")
    print("=" * 50)

    from src.flare_event_generator import FlareEventGenerator

    generation_config = config.get('generation', {})
    test_mode = config.get('test_mode', False)

    if test_mode:
        total_sequences = generation_config.get('num_test_sequences', 50)
        print(f"🧪 TEST MODE: Will generate {total_sequences} test sequences")
    else:
        num_train = generation_config.get('num_train_sequences', 10)
        num_val = generation_config.get('num_val_sequences', 5)
        total_sequences = num_train + num_val
        print(f"Will generate {total_sequences} flare sequences ({num_train} train + {num_val} val)")

    flare_generator = FlareEventGenerator(config)

    generated_files = flare_generator.generate_batch(total_sequences)

    print(f"\n✅ Step 1 Complete: Generated {len(generated_files)} flare event files")
    # print(f"   Output directory: {flare_generator.output_dir}")

    return generated_files

def run_step2_event_composition(config):
    print("\n🚀 Step 2: Event Composition")
    print("=" * 50)
    
    flare_events_dir = os.path.join('output', 'data', 'flare_events')
    if not os.path.exists(flare_events_dir) or not os.listdir(flare_events_dir):
        print(f"❌ Error: No flare events found in {flare_events_dir}")
        print("   Please run Step 1 first with: python main.py --step 1")
        return [], []
    
    event_composer = EventComposer(config)
    
    bg_files, merge_files = event_composer.compose_batch()
    
    print(f"\n✅ Step 2 Complete: Generated {len(bg_files)} background + {len(merge_files)} merged event files")
    
    for method_name, paths in event_composer.output_dirs.items():
        print(f"   {method_name} method:")
        print(f"     - Stage 1 (BG+Light): {paths['stage1']}")
        print(f"     - Stage 2 (Full Scene): {paths['stage2']}")
    
    return bg_files, merge_files

def run_both_steps(config):
    print("🚀 EventMamba-FX Two-Step Event Generator")
    print("=" * 60)
    
    flare_files = run_step1_flare_generation(config)
    
    if not flare_files:
        print("❌ Step 1 failed, stopping pipeline")
        return
    
    bg_files, merge_files = run_step2_event_composition(config)
    
    print(f"\n🎉 Complete Pipeline Success!")
    print(f"   Flare events: {len(flare_files)} files")
    print(f"   Stage 1 (BG+Light): {len(bg_files)} files") 
    print(f"   Stage 2 (Full Scene): {len(merge_files)} files")
    print(f"   Total processing complete.")

def main(config, step=None):
    
    if step == 1:
        run_step1_flare_generation(config)
    elif step == 2:
        run_step2_event_composition(config)
    else:
        run_both_steps(config)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="EventMamba-FX Two-Step Event Generator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py                              # Run complete pipeline (Step 1 + Step 2)
  python main.py --step 1                     # Only generate flare events
  python main.py --step 2                     # Only compose events (requires Step 1 first)
  python main.py --debug                      # Run with debug visualizations
  python main.py --step 1 --debug             # Generate flare events with debug
  python main.py --test                       # Run in test mode (default 10 sequences)
  python main.py --test --debug               # Test mode with debug visualizations
  python main.py --test --num-sequences 50    # Test mode with 50 sequences
  python main.py --test --step 2 --num-sequences 5  # Only Step 2, 5 sequences
        """
    )

    parser.add_argument('--config', type=str, default='configs/config.yaml',
                       help="Path to the YAML configuration file.")
    parser.add_argument('--step', type=int, choices=[1, 2],
                       help="Run specific step: 1=flare generation, 2=event composition")
    parser.add_argument('--debug', action='store_true',
                       help="Enable debug mode with visualizations.")
    parser.add_argument('--test', action='store_true',
                       help="Run in test mode: generate small test dataset with fixed seed.")
    parser.add_argument('--num-sequences', type=int, default=None,
                       help="Number of sequences to generate in test mode (overrides config default).")
    parser.add_argument('--white-bg', action='store_true',
                       help="Force white background for visualizations (debug mode already defaults to white).")
    parser.add_argument('--flare-position', type=str, choices=['upper', 'center', 'random'], default=None,
                       help="Flare position mode: 'upper' (test mode default, top 1/3), 'center' (image center), 'random' (full frame). Only affects test mode.")

    args = parser.parse_args()

    # Load configuration
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    # Configure test mode
    if args.test:
        config['test_mode'] = True
        print("🧪 Test mode enabled - generating test dataset")

        # Apply test mode settings
        generation_config = config.setdefault('generation', {})
        test_seed = generation_config.get('test_mode_seed', 42)

        # Set fixed random seed for reproducibility
        import random
        import numpy as np
        random.seed(test_seed)
        np.random.seed(test_seed)
        print(f"   Random seed: {test_seed} (reproducible)")

        # Force physics_noRandom_noTen for test mode
        composition_config = config.setdefault('composition', {})
        composition_config['merge_method'] = 'physics_noRandom_noTen'
        composition_config['generate_both_methods'] = False
        print(f"   Merge method: physics_noRandom_noTen (no time jitter)")

        # 🆕 Configure flare position mode
        if args.flare_position is not None:
            config['flare_position_mode'] = args.flare_position
            print(f"   Flare position: {args.flare_position} (from command line)")
        else:
            config['flare_position_mode'] = generation_config.get('test_flare_position_mode', 'upper')
            print(f"   Flare position: {config['flare_position_mode']} (default)")

        # Override output paths to test/ directory
        if args.num_sequences is not None:
            test_num = args.num_sequences
            generation_config['num_test_sequences'] = test_num
            print(f"   Test sequences: {test_num} (from command line)")
        else:
            test_num = generation_config.get('num_test_sequences', 10)
            print(f"   Test sequences: {test_num} (from config)")

    # Configure debug mode
    if args.debug:
        config['debug_mode'] = True
        print("🔍 Debug mode enabled - visualizations will be saved")

        if not args.test:
            # Reduce sequences for debug mode (only if not in test mode)
            generation_config = config.setdefault('generation', {})
            generation_config['num_train_sequences'] = generation_config.get('debug_sequences', 3)
            generation_config['num_val_sequences'] = generation_config.get('debug_sequences', 2)
            print(f"   Debug sequences: {generation_config['num_train_sequences']} train + {generation_config['num_val_sequences']} val")

    # 🆕 Configure visualization background color
    if args.white_bg:
        config['visualization_background'] = 'white'
        print("🎨 Visualization background: white (explicitly set)")
    elif args.debug:
        config['visualization_background'] = 'white'
        print("🎨 Visualization background: white (debug mode default, paper-friendly)")
    else:
        config['visualization_background'] = 'black'

    # Run the specified step(s)
    main(config, step=args.step)