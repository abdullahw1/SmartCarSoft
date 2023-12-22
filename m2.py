import sys
import getopt
from core import core
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import ConsoleSpanExporter
from opentelemetry.sdk.resources import Resource
import logging

# Initialize tracer provider
trace.set_tracer_provider(TracerProvider(resource=Resource.create().merge(default=True)))
tracer = trace.get_tracer(__name__)

# Configure loggers
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main(argv):
    video_path = "./Videos/Muneeb iphone 6.mp4"
    tm = False  # flag for momentary time of collusion
    rt = True  # real-time flag

    options = "acr"
    long_options = ["acceleration", "camera", "realtime"]

    try:
        arguments, values = getopt.getopt(argv, options, long_options)

        for currentArgument, currentValue in arguments:

            if currentArgument in ("-a", "--acceleration"):
                logger.info("Displaying Momentary Time to Contact without Acceleration")
                tm = True

            elif currentArgument in ("-c", "--camera"):
                logger.info("Using camera")
                video_path = False

            elif currentArgument in ("-r", "--realtime"):
                logger.info("Not using real time")
                rt = False

    except getopt.error as err:
        logger.error(str(err))

    # Call the core functionality from core.py within a trace
    with tracer.start_as_current_span("core-processing"):
        c = core()
        c.system(video_path, "/media/dev/UUI/detection.avi", input_size=320, show=True, iou_threshold=0.1,
                 rectangle_colors=(255, 0, 0), Track_only=['car', 'truck', 'motorbike', 'person'], display_tm=tm,
                 realTime=rt)


if __name__ == "__main__":
    # Create a SpanExporter for console output
    exporter = ConsoleSpanExporter()
    span_processor = BatchSpanProcessor(exporter)

    # Register the span processor with the tracer
    trace.get_tracer_provider().add_span_processor(span_processor)

    main(sys.argv[1:])

