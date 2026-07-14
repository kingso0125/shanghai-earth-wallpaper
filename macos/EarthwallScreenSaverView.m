#import <ScreenSaver/ScreenSaver.h>

@interface EarthwallScreenSaverView : ScreenSaverView
@property(nonatomic, strong) NSImage *earthImage;
@property(nonatomic, strong) NSDate *loadedAt;
@end

@implementation EarthwallScreenSaverView

- (instancetype)initWithFrame:(NSRect)frame isPreview:(BOOL)isPreview {
    self = [super initWithFrame:frame isPreview:isPreview];
    if (self) {
        self.animationTimeInterval = 30.0;
        [self reloadImage];
    }
    return self;
}

- (NSString *)wallpaperPath {
    return [NSHomeDirectory() stringByAppendingPathComponent:
        @"Library/Application Support/EarthwallMac/current/mac-lock.jpg"];
}

- (void)reloadImage {
    self.earthImage = [[NSImage alloc] initWithContentsOfFile:[self wallpaperPath]];
    self.loadedAt = [NSDate date];
    [self setNeedsDisplay:YES];
}

- (void)animateOneFrame {
    [self reloadImage];
}

- (void)drawRect:(NSRect)rect {
    [[NSColor blackColor] setFill];
    NSRectFill(self.bounds);
    if (!self.earthImage) return;

    NSSize imageSize = self.earthImage.size;
    CGFloat scale = MIN(NSWidth(self.bounds) / imageSize.width,
                        NSHeight(self.bounds) / imageSize.height);
    NSSize drawSize = NSMakeSize(imageSize.width * scale, imageSize.height * scale);
    NSRect destination = NSMakeRect(
        NSMidX(self.bounds) - drawSize.width / 2.0,
        NSMidY(self.bounds) - drawSize.height / 2.0,
        drawSize.width,
        drawSize.height
    );
    [self.earthImage drawInRect:destination
                       fromRect:NSZeroRect
                      operation:NSCompositingOperationSourceOver
                       fraction:1.0
                 respectFlipped:YES
                          hints:@{NSImageHintInterpolation: @(NSImageInterpolationHigh)}];
}

- (BOOL)hasConfigureSheet { return NO; }

@end
