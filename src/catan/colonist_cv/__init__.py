"""Computer-vision integration scaffolding for live Colonist board reading."""

from .advisor import ActionAdvice, HeuristicActionAdvisor
from .bootstrap import AutoBoardBootstrap, TokenCandidate, auto_bootstrap_board
from .context_ocr import ScreenContextDetection, detect_hand_resources, detect_local_player_color, read_screen_context
from .detector import (
    ColonistVisionDetector,
    DefaultPlayerPalette,
    DefaultResourcePalette,
    DetectionError,
    PrototypeColorClassifier,
)
from .geometry import BoardCalibration, estimate_homography
from .opening_live import OpeningLiveRunner, OpeningScreenAnalysis, OpeningSuggestion, PromptKind, ScreenPrompt, analyze_opening_screen
from .ocr import OCRUnavailableError, easyocr_available
from .runtime import LiveAdvisorRunner, LiveContext, ScreenRegion, apply_context_overrides, load_live_context
from .schema import PlayerColor, PrivateObservation, PublicStructures, VisionFrameObservation
from .tracker import ColonistVisionTracker, build_state_from_observation

__all__ = [
    "ActionAdvice",
    "AutoBoardBootstrap",
    "BoardCalibration",
    "ColonistVisionDetector",
    "ColonistVisionTracker",
    "DefaultPlayerPalette",
    "DefaultResourcePalette",
    "DetectionError",
    "ScreenContextDetection",
    "HeuristicActionAdvisor",
    "OCRUnavailableError",
    "OpeningLiveRunner",
    "OpeningScreenAnalysis",
    "OpeningSuggestion",
    "PromptKind",
    "ScreenPrompt",
    "TokenCandidate",
    "LiveAdvisorRunner",
    "LiveContext",
    "PlayerColor",
    "PrivateObservation",
    "PrototypeColorClassifier",
    "PublicStructures",
    "ScreenRegion",
    "VisionFrameObservation",
    "apply_context_overrides",
    "analyze_opening_screen",
    "auto_bootstrap_board",
    "build_state_from_observation",
    "detect_hand_resources",
    "detect_local_player_color",
    "easyocr_available",
    "estimate_homography",
    "load_live_context",
    "read_screen_context",
]
