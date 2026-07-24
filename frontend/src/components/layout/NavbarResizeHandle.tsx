import { useCallback, useEffect, useRef } from "react";
import {
  NAVBAR_MAX_WIDTH,
  NAVBAR_MIN_WIDTH,
} from "@/store/settingsStore";

interface NavbarResizeHandleProps {
  width: number;
  onWidthChange: (width: number) => void;
}

export function NavbarResizeHandle({
  width,
  onWidthChange,
}: NavbarResizeHandleProps) {
  const draggingRef = useRef(false);
  const startXRef = useRef(0);
  const startWidthRef = useRef(width);

  const endDrag = useCallback(() => {
    if (!draggingRef.current) return;
    draggingRef.current = false;
    document.body.style.cursor = "";
    document.body.style.userSelect = "";
  }, []);

  const onPointerDown = useCallback(
    (clientX: number) => {
      draggingRef.current = true;
      startXRef.current = clientX;
      startWidthRef.current = width;
      document.body.style.cursor = "col-resize";
      document.body.style.userSelect = "none";
    },
    [width],
  );

  useEffect(() => {
    const onPointerMove = (clientX: number) => {
      if (!draggingRef.current) return;
      const delta = clientX - startXRef.current;
      onWidthChange(startWidthRef.current + delta);
    };

    const handleMouseMove = (e: MouseEvent) => onPointerMove(e.clientX);
    const handleTouchMove = (e: TouchEvent) => {
      if (e.touches[0]) onPointerMove(e.touches[0].clientX);
    };
    const handleMouseUp = () => endDrag();
    const handleTouchEnd = () => endDrag();

    window.addEventListener("mousemove", handleMouseMove);
    window.addEventListener("mouseup", handleMouseUp);
    window.addEventListener("touchmove", handleTouchMove);
    window.addEventListener("touchend", handleTouchEnd);

    return () => {
      window.removeEventListener("mousemove", handleMouseMove);
      window.removeEventListener("mouseup", handleMouseUp);
      window.removeEventListener("touchmove", handleTouchMove);
      window.removeEventListener("touchend", handleTouchEnd);
    };
  }, [endDrag, onWidthChange]);

  return (
    <div
      role="separator"
      aria-orientation="vertical"
      aria-label="Resize sidebar"
      aria-valuenow={width}
      aria-valuemin={NAVBAR_MIN_WIDTH}
      aria-valuemax={NAVBAR_MAX_WIDTH}
      onMouseDown={(e) => {
        e.preventDefault();
        onPointerDown(e.clientX);
      }}
      onTouchStart={(e) => {
        onPointerDown(e.touches[0].clientX);
      }}
      style={{
        position: "absolute",
        top: 0,
        right: 0,
        width: 6,
        height: "100%",
        cursor: "col-resize",
        zIndex: 10,
        touchAction: "none",
      }}
    />
  );
}
