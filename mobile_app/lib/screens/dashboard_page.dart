import 'package:flutter/material.dart';
import '../theme/app_colors.dart';
import 'chat_page.dart';
import 'profile_page.dart';
import 'resources_page.dart';
import 'risk_report_page.dart';

class DashboardPage extends StatefulWidget {
  const DashboardPage({super.key});

  @override
  State<DashboardPage> createState() => _DashboardPageState();
}

class _DashboardPageState extends State<DashboardPage> {
  int _selectedTab = 0;

  static const _activities = [
    ('Oct 24, 2023', 'Morning Session', 'Stressed', Color(0xFFE65F73)),
    ('Oct 23, 2023', 'Evening Review', 'Anxious', Color(0xFFFFA21A)),
    ('Oct 22, 2023', 'Daily Check-in', 'Normal', Color(0xFF48C78E)),
  ];

  void _openFeature(String name) {
    ScaffoldMessenger.of(context)
      ..hideCurrentSnackBar()
      ..showSnackBar(SnackBar(content: Text('$name is coming soon.'), behavior: SnackBarBehavior.floating));
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: const Color(0xFFF4F3FF),
      body: SafeArea(
        child: Center(
          child: ConstrainedBox(
            constraints: const BoxConstraints(maxWidth: 480),
            child: Column(
              children: [
                Expanded(
                  child: _selectedTab == 1
                      ? const ChatPage()
                      : _selectedTab == 4
                      ? const ProfilePage()
                      : _selectedTab == 2
                          ? RiskReportPage(onBack: () => setState(() => _selectedTab = 0))
                          : _selectedTab == 3
                              ? const ResourcesPage()
                      : SingleChildScrollView(
                    padding: const EdgeInsets.fromLTRB(24, 20, 24, 16),
                    child: Column(
                      crossAxisAlignment: CrossAxisAlignment.start,
                      children: [
                        Row(
                          children: [
                            const Expanded(child: Text('Good Morning, Kamal 👋', style: TextStyle(color: AppColors.ink, fontSize: 14, fontWeight: FontWeight.w800))),
                            IconButton.filledTonal(
                              onPressed: () => _openFeature('Notifications'),
                              icon: const Icon(Icons.notifications_none_rounded, size: 18),
                              style: IconButton.styleFrom(backgroundColor: Colors.white, foregroundColor: AppColors.ink, fixedSize: const Size(40, 40)),
                            ),
                          ],
                        ),
                        const SizedBox(height: 10),
                        _scoreCard(),
                        const SizedBox(height: 16),
                        GridView.count(
                          crossAxisCount: 2,
                          shrinkWrap: true,
                          physics: const NeverScrollableScrollPhysics(),
                          crossAxisSpacing: 12,
                          mainAxisSpacing: 12,
                          childAspectRatio: 1.3,
                          children: [
                            _DashboardTile(icon: Icons.smart_toy_outlined, label: 'Chatbot', tint: const Color(0xFFEDEBFF), iconColor: const Color(0xFF8174E8), onTap: () => setState(() => _selectedTab = 1)),
                            _DashboardTile(icon: Icons.edit_note_rounded, label: 'Journal', tint: const Color(0xFFE7F0FF), iconColor: const Color(0xFF5590F2), onTap: () => _openFeature('Journal')),
                            _DashboardTile(icon: Icons.insert_chart_outlined_rounded, label: 'Analytics', tint: const Color(0xFFFFF0E1), iconColor: const Color(0xFFFF8E35), onTap: () => setState(() => _selectedTab = 2)),
                            _DashboardTile(icon: Icons.menu_book_rounded, label: 'Resources', tint: const Color(0xFFE2F9EA), iconColor: const Color(0xFF42C783), onTap: () => setState(() => _selectedTab = 3)),
                          ],
                        ),
                        const SizedBox(height: 19),
                        Row(
                          children: [
                            const Expanded(child: Text('Recent Activity', style: TextStyle(color: AppColors.ink, fontSize: 13, fontWeight: FontWeight.w800))),
                            TextButton(onPressed: () => _openFeature('Activity history'), style: TextButton.styleFrom(minimumSize: Size.zero, padding: const EdgeInsets.symmetric(horizontal: 4)), child: const Text('See All', style: TextStyle(color: AppColors.purple, fontSize: 10))),
                          ],
                        ),
                        const SizedBox(height: 5),
                        ..._activities.map((activity) => _ActivityCard(activity: activity)),
                      ],
                    ),
                  ),
                ),
                _bottomBar(),
              ],
            ),
          ),
        ),
      ),
    );
  }

  Widget _scoreCard() => Container(
        width: double.infinity,
        padding: const EdgeInsets.symmetric(vertical: 12),
        decoration: BoxDecoration(
          borderRadius: BorderRadius.circular(13),
          gradient: const LinearGradient(colors: [Color(0xFF8275E7), Color(0xFF342C79)], begin: Alignment.topLeft, end: Alignment.bottomRight),
          boxShadow: [BoxShadow(color: const Color(0xFF5146A6).withValues(alpha: .2), blurRadius: 14, offset: const Offset(0, 6))],
        ),
        child: Column(
          children: [
            SizedBox(
              width: 92,
              height: 92,
              child: Stack(
                alignment: Alignment.center,
                children: [
                  SizedBox(width: 88, height: 88, child: CircularProgressIndicator(value: .72, strokeWidth: 5, backgroundColor: Colors.white.withValues(alpha: .3), valueColor: const AlwaysStoppedAnimation(Colors.white))),
                  const Column(mainAxisSize: MainAxisSize.min, children: [Text('72', style: TextStyle(color: Colors.white, fontSize: 23, height: 1, fontWeight: FontWeight.w800)), SizedBox(height: 3), Text('Score', style: TextStyle(color: Colors.white70, fontSize: 8))]),
                ],
              ),
            ),
            const SizedBox(height: 5),
            Container(
              padding: const EdgeInsets.symmetric(horizontal: 9, vertical: 4),
              decoration: BoxDecoration(color: Colors.white.withValues(alpha: .16), borderRadius: BorderRadius.circular(20)),
              child: const Row(mainAxisSize: MainAxisSize.min, children: [Icon(Icons.warning_amber_rounded, color: Colors.white, size: 10), SizedBox(width: 3), Text('High Risk', style: TextStyle(color: Colors.white, fontSize: 9, fontWeight: FontWeight.w600))]),
            ),
            const SizedBox(height: 7),
            const Text("Based on today's analysis", style: TextStyle(color: Colors.white70, fontSize: 9)),
          ],
        ),
      );

  Widget _bottomBar() {
    const items = [
      (Icons.home_rounded, 'Home'),
      (Icons.chat_bubble_outline_rounded, 'Chat'),
      (Icons.insert_chart_outlined_rounded, 'Analytics'),
      (Icons.menu_book_rounded, 'Resources'),
      (Icons.person_outline_rounded, 'Profile'),
    ];
    return Container(
      decoration: const BoxDecoration(color: Colors.white, border: Border(top: BorderSide(color: Color(0xFFEDECF4)))),
      padding: const EdgeInsets.only(top: 8, bottom: 5),
      child: Row(
        mainAxisAlignment: MainAxisAlignment.spaceAround,
        children: List.generate(items.length, (index) {
          final selected = _selectedTab == index;
          return InkWell(
          onTap: () => setState(() => _selectedTab = index),
            borderRadius: BorderRadius.circular(10),
            child: SizedBox(
              width: 56,
              child: Column(
                mainAxisSize: MainAxisSize.min,
                children: [
                  Icon(items[index].$1, size: 18, color: selected ? AppColors.purple : const Color(0xFF9998A8)),
                  const SizedBox(height: 3),
                  Text(items[index].$2, style: TextStyle(fontSize: 8, color: selected ? AppColors.purple : const Color(0xFF9998A8), fontWeight: selected ? FontWeight.w600 : FontWeight.w400)),
                ],
              ),
            ),
          );
        }),
      ),
    );
  }
}

class _DashboardTile extends StatelessWidget {
  const _DashboardTile({required this.icon, required this.label, required this.tint, required this.iconColor, required this.onTap});

  final IconData icon;
  final String label;
  final Color tint;
  final Color iconColor;
  final VoidCallback onTap;

  @override
  Widget build(BuildContext context) => Material(
        color: Colors.white,
        borderRadius: BorderRadius.circular(13),
        child: InkWell(
          onTap: onTap,
          borderRadius: BorderRadius.circular(13),
          child: Column(
            mainAxisAlignment: MainAxisAlignment.center,
            children: [
              Container(width: 34, height: 34, decoration: BoxDecoration(color: tint, borderRadius: BorderRadius.circular(10)), child: Icon(icon, size: 19, color: iconColor)),
              const SizedBox(height: 7),
              Text(label, style: const TextStyle(color: AppColors.ink, fontSize: 10, fontWeight: FontWeight.w600)),
            ],
          ),
        ),
      );
}

class _ActivityCard extends StatelessWidget {
  const _ActivityCard({required this.activity});

  final (String, String, String, Color) activity;

  @override
  Widget build(BuildContext context) => Container(
        margin: const EdgeInsets.only(bottom: 8),
        padding: const EdgeInsets.symmetric(horizontal: 11, vertical: 9),
        decoration: BoxDecoration(color: Colors.white, borderRadius: BorderRadius.circular(11)),
        child: Row(
          children: [
            Container(width: 28, height: 28, decoration: BoxDecoration(color: const Color(0xFFF6F5FA), borderRadius: BorderRadius.circular(8)), child: const Icon(Icons.calendar_today_outlined, color: Color(0xFF9998A8), size: 13)),
            const SizedBox(width: 9),
            Expanded(child: Column(crossAxisAlignment: CrossAxisAlignment.start, children: [Text(activity.$1, style: const TextStyle(color: AppColors.ink, fontSize: 9, fontWeight: FontWeight.w700)), const SizedBox(height: 2), Text(activity.$2, style: const TextStyle(color: Color(0xFF9998A8), fontSize: 8))])),
            Container(width: 6, height: 6, decoration: BoxDecoration(color: activity.$4, shape: BoxShape.circle)),
            const SizedBox(width: 5),
            Text(activity.$3, style: TextStyle(color: activity.$4, fontSize: 9, fontWeight: FontWeight.w600)),
          ],
        ),
      );
}
