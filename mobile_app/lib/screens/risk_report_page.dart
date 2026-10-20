import 'package:flutter/material.dart';
import '../theme/app_colors.dart';

class RiskReportPage extends StatefulWidget {
  const RiskReportPage({super.key, required this.onBack});

  final VoidCallback onBack;

  @override
  State<RiskReportPage> createState() => _RiskReportPageState();
}

class _RiskReportPageState extends State<RiskReportPage> {
  int _period = 0;
  static const _red = Color(0xFFF44343);
  static const _orange = Color(0xFFFFA321);
  static const _green = Color(0xFF16C784);

  void _message(String text) {
    ScaffoldMessenger.of(context)
      ..hideCurrentSnackBar()
      ..showSnackBar(SnackBar(content: Text(text), behavior: SnackBarBehavior.floating));
  }

  @override
  Widget build(BuildContext context) => SingleChildScrollView(
        padding: const EdgeInsets.fromLTRB(24, 16, 24, 18),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Row(
              children: [
                IconButton(onPressed: widget.onBack, icon: const Icon(Icons.arrow_back_ios_new_rounded, size: 17, color: AppColors.ink), padding: EdgeInsets.zero, constraints: const BoxConstraints(minWidth: 28, minHeight: 36)),
                const Expanded(child: Center(child: Text('Your Risk Report', style: TextStyle(color: AppColors.ink, fontSize: 13, fontWeight: FontWeight.w800)))),
                const SizedBox(width: 28),
              ],
            ),
            const SizedBox(height: 10),
            Container(
              padding: const EdgeInsets.all(4),
              decoration: BoxDecoration(color: Colors.white, borderRadius: BorderRadius.circular(10)),
              child: Row(children: List.generate(3, (index) {
                const labels = ['Today', 'This Week', 'This Month'];
                final selected = _period == index;
                return Expanded(
                  child: GestureDetector(
                    onTap: () => setState(() => _period = index),
                    child: Container(
                      padding: const EdgeInsets.symmetric(vertical: 8),
                      decoration: BoxDecoration(color: selected ? const Color(0xFFF0EEFF) : Colors.transparent, borderRadius: BorderRadius.circular(8)),
                      child: Center(child: Text(labels[index], style: TextStyle(color: selected ? AppColors.purple : const Color(0xFF8E8C9D), fontSize: 9, fontWeight: selected ? FontWeight.w700 : FontWeight.w500))),
                    ),
                  ),
                );
              })),
            ),
            const SizedBox(height: 12),
            Container(
              width: double.infinity,
              padding: const EdgeInsets.symmetric(vertical: 14),
              decoration: BoxDecoration(color: Colors.white, borderRadius: BorderRadius.circular(13)),
              child: Column(
                children: [
                  SizedBox(
                    width: 116,
                    height: 116,
                    child: Stack(alignment: Alignment.center, children: [
                      const SizedBox(width: 112, height: 112, child: CircularProgressIndicator(value: .72, strokeWidth: 9, backgroundColor: Color(0xFFF0F0F3), valueColor: AlwaysStoppedAnimation(_red))),
                      const Column(mainAxisSize: MainAxisSize.min, children: [Text('72', style: TextStyle(color: AppColors.ink, fontSize: 25, height: 1, fontWeight: FontWeight.w800)), SizedBox(height: 4), Text('High Risk', style: TextStyle(color: _red, fontSize: 9, fontWeight: FontWeight.w500))]),
                    ]),
                  ),
                  const SizedBox(height: 13),
                  const Row(mainAxisAlignment: MainAxisAlignment.center, children: [_RiskLegend(color: _green, label: 'Normal'), SizedBox(width: 17), _RiskLegend(color: _orange, label: 'Moderate'), SizedBox(width: 17), _RiskLegend(color: _red, label: 'High Risk')]),
                ],
              ),
            ),
            const SizedBox(height: 15),
            const Text('Metric Breakdown', style: TextStyle(color: AppColors.ink, fontSize: 12, fontWeight: FontWeight.w800)),
            const SizedBox(height: 8),
            const _MetricCard(icon: Icons.psychology_alt_rounded, title: 'Sentiment Analysis', value: '85%', color: _orange, progress: .85),
            const SizedBox(height: 7),
            const _MetricCard(icon: Icons.face_retouching_natural_rounded, title: 'Facial Emotion', value: '40%', color: _green, progress: .40),
            const SizedBox(height: 7),
            const _MetricCard(icon: Icons.record_voice_over_rounded, title: 'Behavioral Pattern', value: '80%', color: _red, progress: .80),
            const SizedBox(height: 10),
            Container(
              width: double.infinity,
              padding: const EdgeInsets.all(13),
              decoration: BoxDecoration(color: const Color(0xFFE4F2FF), borderRadius: BorderRadius.circular(12)),
              child: Column(crossAxisAlignment: CrossAxisAlignment.start, children: [
                const Row(children: [Icon(Icons.lightbulb_outline_rounded, color: AppColors.purple, size: 15), SizedBox(width: 6), Text('Recommendation', style: TextStyle(color: AppColors.ink, fontSize: 11, fontWeight: FontWeight.w800))]),
                const SizedBox(height: 8),
                _recommendation('Take a 10-minute mindful walk to clear your thoughts.'),
                _recommendation('Try a guided 4-7-8 breathing exercise to reduce anxiety.'),
                _recommendation('Limit screen time for the next hour for better mental focus.'),
                const SizedBox(height: 10),
                SizedBox(
                  width: double.infinity,
                  height: 35,
                  child: OutlinedButton.icon(
                    onPressed: () => _message('Counselor booking is coming soon.'),
                    icon: const Icon(Icons.person_outline_rounded, size: 13),
                    label: const Text('Talk to a Counselor', style: TextStyle(fontSize: 9, fontWeight: FontWeight.w600)),
                    style: OutlinedButton.styleFrom(foregroundColor: AppColors.purple, side: const BorderSide(color: AppColors.purple), shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(8))),
                  ),
                ),
              ]),
            ),
          ],
        ),
      );

  Widget _recommendation(String text) => Padding(
        padding: const EdgeInsets.only(bottom: 6),
        child: Row(crossAxisAlignment: CrossAxisAlignment.start, children: [const Padding(padding: EdgeInsets.only(top: 5), child: Icon(Icons.circle, size: 4, color: AppColors.purple)), const SizedBox(width: 7), Expanded(child: Text(text, style: const TextStyle(color: Color(0xFF575477), fontSize: 9, height: 1.35)))]),
      );
}

class _RiskLegend extends StatelessWidget {
  const _RiskLegend({required this.color, required this.label});

  final Color color;
  final String label;

  @override
  Widget build(BuildContext context) => Row(mainAxisSize: MainAxisSize.min, children: [Container(width: 5, height: 5, decoration: BoxDecoration(color: color, shape: BoxShape.circle)), const SizedBox(width: 4), Text(label, style: const TextStyle(color: Color(0xFF85849A), fontSize: 7))]);
}

class _MetricCard extends StatelessWidget {
  const _MetricCard({required this.icon, required this.title, required this.value, required this.color, required this.progress});

  final IconData icon;
  final String title;
  final String value;
  final Color color;
  final double progress;

  @override
  Widget build(BuildContext context) => Container(
        padding: const EdgeInsets.fromLTRB(11, 9, 11, 10),
        decoration: BoxDecoration(color: Colors.white, borderRadius: BorderRadius.circular(11)),
        child: Column(children: [
          Row(children: [Container(width: 22, height: 22, decoration: BoxDecoration(color: const Color(0xFFF0EEFF), borderRadius: BorderRadius.circular(7)), child: Icon(icon, color: AppColors.purple, size: 13)), const SizedBox(width: 7), Expanded(child: Text(title, style: const TextStyle(color: AppColors.ink, fontSize: 9, fontWeight: FontWeight.w600))), Text(value, style: TextStyle(color: color, fontSize: 9, fontWeight: FontWeight.w700))]),
          const SizedBox(height: 7),
          ClipRRect(borderRadius: BorderRadius.circular(10), child: LinearProgressIndicator(value: progress, minHeight: 4, backgroundColor: const Color(0xFFF0F0F3), valueColor: AlwaysStoppedAnimation(color))),
        ]),
      );
}
