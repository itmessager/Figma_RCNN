try:
    import cpu_nms as nms
except ImportError:
    from detection.snms.standard_nms import nms
